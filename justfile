DIR := justfile_directory()
NEO4J_DIR := DIR + "/neo4j/data"




# project initialization
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

# end project initialization


# start frontend and backend
# start:
#     cd server && cargo run
# end start frontend and backend



# front
front:
    trunck serve

# end front


# backend(server)
server:
    cargo run -p platform




# ----------------------------------------------------------------
dbg:
    echo {{DIR}}
    echo {{NEO4J_DIR}}
