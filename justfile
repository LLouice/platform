DIR := justfile_directory()
NEO4J_DIR := DIR + "/neo4j/data"




init: init_neo4j redis


redis:
    redis-server


init_neo4j: start_docker
    docker run \
        -d \
        --publish=7474:7474 --publish=7687:7687 \
        --volume={{NEO4J_DIR}}:/data \
        -e NEO4J_AUTH=neo4j/symptom \
        neo4j

start_docker:
    sudo systemctl start docker


dbg:
    echo {{DIR}}
    echo {{NEO4J_DIR}}
