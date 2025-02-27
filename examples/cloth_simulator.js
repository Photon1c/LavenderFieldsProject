import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import Ammo from 'ammo.js';

let scene, camera, renderer, controls;
let cloth, clothMesh, physicsWorld;
const clothWidth = 10, clothHeight = 10, numSegments = 20;
let softBody;

async function initPhysics() {
    await Ammo();
    physicsWorld = new Ammo.btSoftRigidDynamicsWorld(
        new Ammo.btDefaultCollisionConfiguration(),
        new Ammo.btDispatcher(),
        new Ammo.btDbvtBroadphase(),
        new Ammo.btSequentialImpulseConstraintSolver(),
        new Ammo.btSoftBodySolver()
    );
    physicsWorld.setGravity(new Ammo.btVector3(0, -9.8, 0));
}

function createScene() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 100);
    camera.position.set(0, 5, 20);
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);
    controls = new OrbitControls(camera, renderer.domElement);
    
    // Lights
    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(10, 10, 10);
    scene.add(light);
}

function createCloth() {
    const geometry = new THREE.PlaneGeometry(clothWidth, clothHeight, numSegments, numSegments);
    const material = new THREE.MeshStandardMaterial({
        color: 0xff4444, side: THREE.DoubleSide, wireframe: true
    });
    clothMesh = new THREE.Mesh(geometry, material);
    clothMesh.rotation.x = -Math.PI / 2;
    scene.add(clothMesh);

    // Ammo.js Soft Body Setup
    const softBodyHelpers = new Ammo.btSoftBodyHelpers();
    const clothCorner = new Ammo.btVector3(-clothWidth / 2, clothHeight / 2, 0);
    const clothResolution = numSegments + 1;
    softBody = softBodyHelpers.CreatePatch(
        physicsWorld.getWorldInfo(),
        clothCorner, clothCorner, clothResolution, clothResolution,
        0, true
    );
    const sbConfig = softBody.get_m_cfg();
    sbConfig.set_viterations(10);
    sbConfig.set_piterations(10);
    softBody.setTotalMass(0.5);
    physicsWorld.addSoftBody(softBody);
}

function updatePhysics(deltaTime) {
    physicsWorld.stepSimulation(deltaTime, 10);
    const nodes = softBody.get_m_nodes();
    const positions = clothMesh.geometry.attributes.position.array;
    let index = 0;
    for (let i = 0; i < positions.length / 3; i++) {
        const node = nodes.at(i);
        const pos = node.get_m_x();
        positions[index++] = pos.x();
        positions[index++] = pos.y();
        positions[index++] = pos.z();
    }
    clothMesh.geometry.attributes.position.needsUpdate = true;
}

function animate() {
    requestAnimationFrame(animate);
    updatePhysics(1 / 60);
    renderer.render(scene, camera);
}

async function main() {
    await initPhysics();
    createScene();
    createCloth();
    animate();
}

main();
