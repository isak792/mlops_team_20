# Instrucciones para Configuración de Docker y DVC

## Levantamiento de Contenedores de Docker

Se utilizaron contenedores de Docker en una máquina virtual en la nube, asegurando que todos los compañeros tengan acceso a los servicios requeridos.

Los servicios que se levantaron fue un servidor de MinIO para simular buckets de S3, y un servidor de MLFlow. Se configuró MinIO para actuar como un 'Artifact Store' para MLFlow, además de usarse como un object storage donde se guardarán todos los archivos del equipo.

Para levantar los contenedores de Docker, se utilizaron los siguientes comandos:
```bash
cd docker
docker-compose up --build -d
```
*Nota*: Esto asume que ya se cuenta con el archivo `.env` en la carpeta de `/docker`. Este no se incluye en el repositorio por motivos de seguridad.


## Configuración de DVC

Se utilizaron los siguientes comandos para inicializar el ambiente DVC:

```bash
dvc init
dvc remote add -d minio-remote s3://data
dvc remote modify minio-remote endpointurl http://24.144.69.175:9000
dvc remote modify minio-remote access_key_id [access-key-id]
dvc remote modify minio-remote secret_access_key [secret-access-key]
```

DVC se configuró para manejar las versiones de los archivos dentro de la carpeta `data/raw`. 

```bash
dvc add data/raw
```

Además, el pipeline de DVC genera datos que automáticamente se suben a la carpeta `data/processed`, que también es manejada por DVC.

Para correr el pipeline, se debe usar este comando:

```bash
dvc repro
```

Los parámetros del pipeline pueden ser modificados en el archivo `params.yaml`