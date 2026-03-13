```mermaid
flowchart TD
    U[User]
    R[Receptionist Agent]
    C[Clerk Agent]
    E[Executive Agent]
    S[Specialist Agent]
    K[Custodian Agent]
    A[Archivist Agent]

    U -->|request / interaction| R
    R -->|create ticket| C

    C -->|strategic / high-scope work| E
    C -->|execution / content work| S
    C -->|maintenance / cleanup / environment| K

    S -->|store results / logs / artifacts| A
    K -->|store logs / maintenance record| A
```
