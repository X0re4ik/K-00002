from src.shared.api import api_qdrant_client, create_collect_if_not_exists

from src.shared.configs import PROJECT_SETTINGS, get_project_logger
from src.models.embedding import get_embedding_size


logger = get_project_logger()


async def init_qdrant_collections():
    _collection = PROJECT_SETTINGS.api.qdrant.text_collection
    logger.info(f"Qdrant create collection: {_collection}")
    create_collect_if_not_exists(
        api_qdrant_client,
        _collection,
        get_embedding_size(),
    )
