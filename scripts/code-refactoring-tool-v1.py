import os
import ast
import re
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple, Optional
import argparse
import datetime
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class DependencyAnalyzer(ast.NodeVisitor):
    """Analyzes Python code to extract function/class dependencies."""
    
    def __init__(self):
        self.functions = {}  # name -> node
        self.classes = {}    # name -> node
        self.imports = []    # list of import nodes
        self.function_calls = defaultdict(set)  # function_name -> set of called functions
        self.class_dependencies = defaultdict(set)  # class_name -> set of used classes
        self.function_class_usage = defaultdict(set)  # function_name -> set of used classes
        self.current_function = None
        self.current_class = None
        self.line_counts = {}  # name -> line count
        
    def visit_FunctionDef(self, node):
        """Extract function definitions and their dependencies."""
        old_function = self.current_function
        self.current_function = node.name
        
        # Store function node
        self.functions[node.name] = node
        
        # Calculate line count
        end_lineno = getattr(node, 'end_lineno', node.lineno + len(node.body))
        self.line_counts[node.name] = end_lineno - node.lineno + 1
        
        # Visit function body to find dependencies
        self.generic_visit(node)
        
        self.current_function = old_function
    
    def visit_ClassDef(self, node):
        """Extract class definitions and their dependencies."""
        old_class = self.current_class
        self.current_class = node.name
        
        # Store class node
        self.classes[node.name] = node
        
        # Calculate line count
        end_lineno = getattr(node, 'end_lineno', node.lineno + len(node.body))
        self.line_counts[node.name] = end_lineno - node.lineno + 1
        
        # Visit class body to find dependencies
        self.generic_visit(node)
        
        self.current_class = old_class
    
    def visit_Call(self, node):
        """Track function calls and class instantiations."""
        # Record function calls
        if isinstance(node.func, ast.Name):
            if self.current_function:
                self.function_calls[self.current_function].add(node.func.id)
            elif self.current_class:
                self.class_dependencies[self.current_class].add(node.func.id)
        
        # Record class instantiations (simplistic approach)
        elif isinstance(node.func, ast.Name) and node.func.id[0].isupper():
            if self.current_function:
                self.function_class_usage[self.current_function].add(node.func.id)
            elif self.current_class:
                self.class_dependencies[self.current_class].add(node.func.id)
        
        self.generic_visit(node)
    
    def visit_Import(self, node):
        """Track import statements."""
        self.imports.append(node)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Track import from statements."""
        self.imports.append(node)
        self.generic_visit(node)

class CodeRefactorer:
    """Handles the code refactoring process."""
    
    def __init__(self, max_lines_per_file=500, output_dir="refactored", backup_dir="backup"):
        self.max_lines_per_file = max_lines_per_file
        self.output_dir = output_dir
        self.backup_dir = backup_dir
        self.report = {
            "original_files": {},
            "refactored_files": {},
            "summary": {}
        }
        
        # Create output directories if they don't exist
        for directory in [output_dir, backup_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def analyze_file(self, file_path: str) -> Tuple[ast.Module, DependencyAnalyzer]:
        """Analyze a Python file to extract its structure and dependencies."""
        logger.info(f"Analyzing file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
            analyzer = DependencyAnalyzer()
            analyzer.visit(tree)
            
            # Store original file information
            self.report["original_files"][file_path] = {
                "functions": list(analyzer.functions.keys()),
                "classes": list(analyzer.classes.keys()),
                "line_count": len(content.splitlines())
            }
            
            return tree, analyzer
            
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {str(e)}")
            raise
    
    def create_dependency_graph(self, analyzer: DependencyAnalyzer) -> nx.DiGraph:
        """Create a directed graph representing dependencies between functions and classes."""
        G = nx.DiGraph()
        
        # Add nodes for functions and classes
        for func_name in analyzer.functions:
            G.add_node(func_name, type="function", size=analyzer.line_counts.get(func_name, 0))
        
        for class_name in analyzer.classes:
            G.add_node(class_name, type="class", size=analyzer.line_counts.get(class_name, 0))
        
        # Add edges for function calls
        for caller, callees in analyzer.function_calls.items():
            for callee in callees:
                if callee in analyzer.functions:
                    G.add_edge(caller, callee, type="function_call")
        
        # Add edges for class dependencies
        for class_name, dependencies in analyzer.class_dependencies.items():
            for dep in dependencies:
                if dep in analyzer.classes:
                    G.add_edge(class_name, dep, type="class_dependency")
        
        # Add edges for function-class usage
        for func_name, used_classes in analyzer.function_class_usage.items():
            for class_name in used_classes:
                if class_name in analyzer.classes:
                    G.add_edge(func_name, class_name, type="uses_class")
        
        return G
    
    def visualize_dependency_graph(self, G: nx.DiGraph, file_name: str):
        """Generate a visualization of the dependency graph."""
        output_file = f"{self.output_dir}/{os.path.basename(file_name)}_dependency_graph.png"
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)  # Positions for all nodes
        
        # Draw nodes
        function_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'function']
        class_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'class']
        
        nx.draw_networkx_nodes(G, pos, nodelist=function_nodes, node_color='lightblue', node_size=500, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=class_nodes, node_color='lightgreen', node_size=700, alpha=0.8)
        
        # Draw edges
        function_call_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'function_call']
        class_dep_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'class_dependency']
        uses_class_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'uses_class']
        
        nx.draw_networkx_edges(G, pos, edgelist=function_call_edges, edge_color='blue', arrows=True)
        nx.draw_networkx_edges(G, pos, edgelist=class_dep_edges, edge_color='green', arrows=True)
        nx.draw_networkx_edges(G, pos, edgelist=uses_class_edges, edge_color='red', arrows=True, style='dashed')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10)
        
        # Add legend
        plt.legend(
            [
                plt.Line2D([0], [0], color='lightblue', marker='o', linestyle=''),
                plt.Line2D([0], [0], color='lightgreen', marker='o', linestyle=''),
                plt.Line2D([0], [0], color='blue', marker='', linestyle='-'),
                plt.Line2D([0], [0], color='green', marker='', linestyle='-'),
                plt.Line2D([0], [0], color='red', marker='', linestyle='--')
            ],
            [
                'Function', 
                'Class',
                'Function Call', 
                'Class Dependency', 
                'Uses Class'
            ],
            loc='upper right'
        )
        
        plt.title(f"Dependency Graph for {os.path.basename(file_name)}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        logger.info(f"Dependency graph saved to {output_file}")
        return output_file
    
    def determine_module_groupings(self, G: nx.DiGraph, analyzer: DependencyAnalyzer) -> List[Dict]:
        """Determine logical groupings of functions and classes for new modules."""
        # Strategy 1: Group by community detection
        communities = list(nx.community.louvain_communities(G.to_undirected()))
        
        # Strategy 2: Group by connected components
        components = list(nx.weakly_connected_components(G))
        
        # Choose the strategy that gives more modules but with balanced sizes
        strategy1_sizes = [len(comm) for comm in communities]
        strategy2_sizes = [len(comp) for comp in components]
        
        if len(communities) > len(components) and max(strategy1_sizes) <= self.max_lines_per_file:
            groupings = [{'items': list(comm), 'type': 'community'} for comm in communities]
        else:
            groupings = [{'items': list(comp), 'type': 'component'} for comp in components]
        
        # Ensure no group is too large
        final_groupings = []
        for group in groupings:
            items = group['items']
            total_lines = sum(analyzer.line_counts.get(item, 0) for item in items)
            
            if total_lines <= self.max_lines_per_file:
                final_groupings.append(group)
            else:
                # Split large groups based on node types (functions vs. classes)
                classes = [item for item in items if item in analyzer.classes]
                functions = [item for item in items if item in analyzer.functions]
                
                # Further split if needed
                class_groups = self._split_by_size(classes, analyzer)
                function_groups = self._split_by_size(functions, analyzer)
                
                for c_group in class_groups:
                    final_groupings.append({'items': c_group, 'type': 'class_group'})
                
                for f_group in function_groups:
                    final_groupings.append({'items': f_group, 'type': 'function_group'})
        
        # Create a sensible naming scheme for modules
        for i, group in enumerate(final_groupings):
            group_items = group['items']
            group_type = group['type']
            
            if group_type == 'class_group' and len(group_items) == 1:
                # Single class gets its own file with lowercase name
                group['module_name'] = group_items[0].lower()
            elif group_type == 'class_group':
                # Multiple classes get descriptive name
                group['module_name'] = f"classes_{i+1}"
            elif group_type == 'function_group':
                # Group functions by common prefix or use generic name
                prefixes = [item.split('_')[0] for item in group_items if '_' in item]
                if prefixes and len(set(prefixes)) == 1:
                    group['module_name'] = f"{prefixes[0]}_utils"
                else:
                    group['module_name'] = f"functions_{i+1}"
            else:
                # Mixed or community-based groups
                classes = [item for item in group_items if item in analyzer.classes]
                if classes:
                    # Name based on the "main" class if there is one
                    group['module_name'] = classes[0].lower()
                else:
                    group['module_name'] = f"module_{i+1}"
        
        return final_groupings
    
    def _split_by_size(self, items: List[str], analyzer: DependencyAnalyzer) -> List[List[str]]:
        """Split a list of items into groups that don't exceed the max line count."""
        groups = []
        current_group = []
        current_size = 0
        
        # Sort by size (descending) to handle larger items first
        sorted_items = sorted(items, key=lambda x: analyzer.line_counts.get(x, 0), reverse=True)
        
        for item in sorted_items:
            item_size = analyzer.line_counts.get(item, 0)
            
            # If item is larger than max size, put it in its own group
            if item_size > self.max_lines_per_file:
                groups.append([item])
                continue
            
            # If adding this item would exceed max size, start a new group
            if current_size + item_size > self.max_lines_per_file:
                if current_group:
                    groups.append(current_group)
                current_group = [item]
                current_size = item_size
            else:
                current_group.append(item)
                current_size += item_size
        
        # Add the last group if it's not empty
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def generate_new_modules(self, file_path: str, tree: ast.Module, analyzer: DependencyAnalyzer, 
                            groupings: List[Dict]) -> Dict[str, str]:
        """Generate new module files based on the determined groupings."""
        logger.info(f"Generating new modules for {file_path}")
        
        original_filename = os.path.basename(file_path)
        base_name = os.path.splitext(original_filename)[0]
        
        # Create a mapping of item name to its AST node
        item_to_node = {**analyzer.functions, **analyzer.classes}
        
        # Map of generated modules
        modules = {}
        
        # Create imports list from original file
        imports_code = self._get_imports_code(tree)
        
        # Track which items are in which modules for cross-referencing
        item_to_module = {}
        
        # Generate package __init__.py
        package_dir = f"{self.output_dir}/{base_name}"
        if not os.path.exists(package_dir):
            os.makedirs(package_dir)
        
        init_imports = []
        
        # First pass: create module files with AST nodes
        for group in groupings:
            module_name = group['module_name']
            module_filename = f"{package_dir}/{module_name}.py"
            
            # Track items in this module
            for item in group['items']:
                item_to_module[item] = module_name
            
            # Initialize module code with imports
            module_code = [
                f"# Generated from {original_filename}",
                f"# Creation date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "# Original imports",
                imports_code,
                ""
            ]
            
            # Add placeholder for cross-module imports
            module_code.append("# Cross-module imports (to be filled in second pass)")
            module_code.append("")
            
            # Add code for each item in the group
            for item_name in group['items']:
                if item_name in item_to_node:
                    node = item_to_node[item_name]
                    item_code = self._get_node_source(file_path, node)
                    
                    # Add provenance comment
                    module_code.append("")
                    module_code.append(f"# From original file: {original_filename}")
                    module_code.append(f"# Original location: around line {node.lineno}")
                    module_code.append(item_code)
                    module_code.append("")
            
            # Store the module code
            modules[module_name] = module_code
            
            # Add to __init__ imports
            init_imports.append(f"from .{module_name} import {', '.join(group['items'])}")
        
        # Second pass: add cross-module imports
        for module_name, module_code in modules.items():
            # Get items in this module
            module_items = [item for item, mod in item_to_module.items() if mod == module_name]
            
            # Find dependencies to other modules
            cross_imports = []
            
            for item in module_items:
                # Function calls to other modules
                if item in analyzer.function_calls:
                    for called_func in analyzer.function_calls[item]:
                        if (called_func in item_to_module and 
                            item_to_module[called_func] != module_name and
                            called_func in analyzer.functions):
                            cross_imports.append((item_to_module[called_func], called_func))
                
                # Class dependencies
                if item in analyzer.class_dependencies:
                    for used_class in analyzer.class_dependencies[item]:
                        if (used_class in item_to_module and 
                            item_to_module[used_class] != module_name and
                            used_class in analyzer.classes):
                            cross_imports.append((item_to_module[used_class], used_class))
                
                # Function-class usage
                if item in analyzer.function_class_usage:
                    for used_class in analyzer.function_class_usage[item]:
                        if (used_class in item_to_module and 
                            item_to_module[used_class] != module_name and
                            used_class in analyzer.classes):
                            cross_imports.append((item_to_module[used_class], used_class))
            
            # Group imports by module
            imports_by_module = defaultdict(set)
            for mod, item in cross_imports:
                imports_by_module[mod].add(item)
            
            # Replace the placeholder with actual imports
            cross_module_imports = []
            for mod, items in imports_by_module.items():
                cross_module_imports.append(f"from .{mod} import {', '.join(sorted(items))}")
            
            # Find the placeholder index
            placeholder_idx = module_code.index("# Cross-module imports (to be filled in second pass)")
            
            # Replace the placeholder
            if cross_module_imports:
                module_code[placeholder_idx] = "# Cross-module imports"
                for i, import_stmt in enumerate(cross_module_imports):
                    module_code.insert(placeholder_idx + 1 + i, import_stmt)
            else:
                module_code[placeholder_idx] = "# No cross-module imports needed"
        
        # Create __init__.py
        init_file = f"{package_dir}/__init__.py"
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write(f"# Package: {base_name}\n")
            f.write(f"# Generated from {original_filename}\n")
            f.write(f"# Creation date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("\n".join(init_imports))
        
        # Write module files
        for module_name, module_code in modules.items():
            module_filename = f"{package_dir}/{module_name}.py"
            with open(module_filename, 'w', encoding='utf-8') as f:
                f.write("\n".join(module_code))
            
            logger.info(f"Created module: {module_filename}")
            
            # Store refactored file information
            module_items = [item for item, mod in item_to_module.items() if mod == module_name]
            functions = [item for item in module_items if item in analyzer.functions]
            classes = [item for item in module_items if item in analyzer.classes]
            
            self.report["refactored_files"][module_filename] = {
                "functions": functions,
                "classes": classes,
                "line_count": len(module_code)
            }
        
        # Add the __init__.py file to the report
        self.report["refactored_files"][init_file] = {
            "type": "package_init",
            "exports": list(item_to_module.keys())
        }
        
        # Return mapping of items to their new modules
        return item_to_module
    
    def _get_imports_code(self, tree: ast.Module) -> str:
        """Extract import statements from the AST."""
        imports = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(ast.unparse(node))
        
        return "\n".join(imports)
    
    def _get_node_source(self, file_path: str, node: ast.AST) -> str:
        """Get the source code for a node from the original file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            file_lines = f.readlines()
        
        # Get end line number (for Python 3.8+ it's available directly)
        if hasattr(node, 'end_lineno'):
            end_lineno = node.end_lineno
        else:
            # For older Python versions, approximate the end line
            end_lineno = node.lineno + len(ast.dump(node).split('\n'))
        
        # Extract the relevant lines
        node_lines = file_lines[node.lineno - 1:end_lineno]
        return ''.join(node_lines)
    
    def create_backup(self, file_path: str):
        """Create a backup of the original file."""
        backup_path = os.path.join(self.backup_dir, os.path.basename(file_path))
        with open(file_path, 'r', encoding='utf-8') as src, open(backup_path, 'w', encoding='utf-8') as dst:
            dst.write(src.read())
        logger.info(f"Created backup: {backup_path}")
        return backup_path
    
    def update_original_file(self, file_path: str, item_to_module: Dict[str, str]):
        """Update the original file with comments indicating where items were moved."""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Create a mapping of line numbers to function/class names
        line_to_item = {}
        
        # Parse the file to get nodes
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        # Identify function and class line numbers
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                line_to_item[node.lineno] = node.name
        
        # Add comments indicating where items were moved
        new_lines = []
        for i, line in enumerate(lines):
            new_lines.append(line)
            line_num = i + 1
            
            if line_num in line_to_item:
                item_name = line_to_item[line_num]
                if item_name in item_to_module:
                    new_module = item_to_module[item_name]
                    comment = f"# REFACTORED: This {type(node).__name__} has been moved to {new_module}.py\n"
                    new_lines.append(comment)
        
        # Write the updated file
        updated_path = f"{self.output_dir}/{os.path.basename(file_path)}"
        with open(updated_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        
        logger.info(f"Updated original file with comments: {updated_path}")
        return updated_path
    
    def generate_report(self):
        """Generate a detailed report of the refactoring process."""
        report_path = f"{self.output_dir}/refactoring_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Code Refactoring Report\n\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Original Files\n\n")
            for file_path, file_info in self.report["original_files"].items():
                f.write(f"### {os.path.basename(file_path)}\n\n")
                f.write(f"- Line count: {file_info['line_count']}\n")
                f.write(f"- Functions: {len(file_info['functions'])}\n")
                f.write(f"- Classes: {len(file_info['classes'])}\n\n")
                
                if file_info['functions']:
                    f.write("Functions:\n")
                    for func in file_info['functions']:
                        f.write(f"- `{func}`\n")
                    f.write("\n")
                
                if file_info['classes']:
                    f.write("Classes:\n")
                    for cls in file_info['classes']:
                        f.write(f"- `{cls}`\n")
                    f.write("\n")
            
            f.write("## Refactored Structure\n\n")
            
            # Group by original file
            refactored_by_original = defaultdict(list)
            for refactored_path, refactored_info in self.report["refactored_files"].items():
                original_file = os.path.basename(refactored_path).split('_')[0]
                refactored_by_original[original_file].append((refactored_path, refactored_info))
            
            for original_file, refactored_files in refactored_by_original.items():
                f.write(f"### Original: {original_file}\n\n")
                f.write("New structure:\n\n")
                f.write("```\n")
                
                # Find package directory
                package_dirs = set()
                for path, _ in refactored_files:
                    package_dir = os.path.dirname(path)
                    if package_dir != self.output_dir:
                        package_dirs.add(os.path.basename(package_dir))
                
                for package in package_dirs:
                    f.write(f"{package}/\n")
                    f.write(f"├── __init__.py\n")
                    
                    # List module files
                    modules = [os.path.basename(path) for path, info in refactored_files 
                              if os.path.dirname(path).endswith(package)]
                    
                    for i, module in enumerate(sorted(modules)):
                        if i == len(modules) - 1:
                            f.write(f"└── {module}\n")
                        else:
                            f.write(f"├── {module}\n")
                
                f.write("```\n\n")
                
                # List content of each module
                for refactored_path, refactored_info in sorted(refactored_files, key=lambda x: x[0]):
                    if "type" in refactored_info and refactored_info["type"] == "package_init":
                        continue
                    
                    module_name = os.path.basename(refactored_path)
                    f.write(f"#### {module_name}\n\n")
                    
                    if "line_count" in refactored_info:
                        f.write(f"- Line count: {refactored_info['line_count']}\n")
                    
                    if "functions" in refactored_info and refactored_info["functions"]:
                        f.write("- Functions:\n")
                        for func in refactored_info["functions"]:
                            f.write(f"  - `{func}`\n")
                    
                    if "classes" in refactored_info and refactored_info["classes"]:
                        f.write("- Classes:\n")
                        for cls in refactored_info["classes"]:
                            f.write(f"  - `{cls}`\n")
                    
                    f.write("\n")
            
            f.write("## Summary\n\n")
            
            # Calculate statistics
            total_original_lines = sum(info["line_count"] for info in self.report["original_files"].values())
            
            refactored_files_with_lines = [info for info in self.report["refactored_files"].values() 
                                          if "line_count" in info]
            total_refactored_lines = sum(info["line_count"] for info in refactored_files_with_lines)
            
            f.write(f"- Original file count: {len(self.report['original_files'])}\n")
            f.write(f"- New module count: {len(self.report['refactored_files'])}\n")
            f.write(f"- Total original lines: {total_original_lines}\n")
            f.write(f"- Total refactored lines: {total_refactored_lines}\n")
            f.write(f"- Size change: {(total_refactored_lines - total_original_lines) / total_original_lines * 100:.1f}%\n")
        
        logger.info(f"Generated report: {report_path}")
        return report_path
    
    def refactor_file(self, file_path: str):
        """Refactor a single file."""
        try:
            # Create backup
            backup_path = self.create_backup(file_path)
            
            # Analyze the file structure
            tree, analyzer = self.analyze_file(file_path)
            
            # Create dependency graph
            dependency_graph = self.create_dependency_graph(analyzer)
            
            # Visualize the graph
            graph_file = self.visualize_dependency_graph(dependency_graph, file_path)
            
            # Determine module groupings
            groupings = self.determine_module_groupings(dependency_graph, analyzer)
            
            # Generate new modules
            item_to_module = self.generate_new_modules(file_path, tree, analyzer, groupings)
            
           
            # Update original file with comments
            updated_file = self.update_original_file(file_path, item_to_module)
            
            # Generate report
            report_path = self.generate_report()
            
            logger.info(f"Successfully refactored {file_path}")
            logger.info(f"Backup saved to: {backup_path}")
            logger.info(f"Updated file saved to: {updated_file}")
            logger.info(f"Detailed report saved to: {report_path}")
            
            return {
                "backup_path": backup_path,
                "updated_file": updated_file,
                "report_path": report_path,
                "graph_file": graph_file,
                "modules": list(item_to_module.values())
            }
            
        except Exception as e:
            logger.error(f"Error refactoring {file_path}: {str(e)}")
            raise

def main():
    """Main function to run the refactoring tool."""
    parser = argparse.ArgumentParser(description="Python Code Refactoring Tool")
    parser.add_argument("file", help="The Python file to refactor")
    parser.add_argument("--max-lines", type=int, default=500, help="Maximum lines per file (default: 500)")
    parser.add_argument("--output-dir", default="refactored", help="Output directory (default: 'refactored')")
    parser.add_argument("--backup-dir", default="backup", help="Backup directory (default: 'backup')")
    
    args = parser.parse_args()
    
    try:
        refactorer = CodeRefactorer(
            max_lines_per_file=args.max_lines,
            output_dir=args.output_dir,
            backup_dir=args.backup_dir
        )
        
        result = refactorer.refactor_file(args.file)
        
        print("\nRefactoring completed successfully!")
        print(f"Backup saved to: {result['backup_path']}")
        print(f"Updated file saved to: {result['updated_file']}")
        print(f"Dependency graph saved to: {result['graph_file']}")
        print(f"Detailed report saved to: {result['report_path']}")
        print(f"New modules created: {len(result['modules'])}")
        
    except Exception as e:
        print(f"Error during refactoring: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())