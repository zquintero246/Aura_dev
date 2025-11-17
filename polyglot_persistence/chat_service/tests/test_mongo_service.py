import importlib
import sys

import mongomock
import pytest


def _reload_mongo_with_mock(monkeypatch):
    """
    Reload the service module so it boots with a Mongomock client instead of pymongo.
    """
    monkeypatch.setattr(
        "pymongo.MongoClient",
        lambda *args, **kwargs: mongomock.MongoClient(),
        raising=False,
    )
    module_name = "polyglot_persistence.chat_service.services.mongo"
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


@pytest.fixture
def mongo_service(monkeypatch):
    return _reload_mongo_with_mock(monkeypatch)


def test_create_and_list_conversation(mongo_service):
    conv = mongo_service.create_conversation(user_id="user-testing", title="Inicio de prueba")
    assert conv["title"] == "Inicio de prueba"
    history = mongo_service.list_conversations("user-testing")
    assert len(history) == 1
    assert history[0]["id"] == conv["id"]


def test_insert_message_and_auto_title(mongo_service):
    conv = mongo_service.create_conversation(user_id="user-testing")
    message = mongo_service.insert_message(
        conversation_id=conv["id"],
        user_id="user-testing",
        role="user",
        content="Necesito ideas para un plan de energía verde por favor",
    )
    assert message["id"]
    history = mongo_service.list_conversations("user-testing")
    assert history[0]["title"].startswith("Ideas para")


def test_insert_message_requires_ownership(mongo_service):
    conv = mongo_service.create_conversation(user_id="owner-id")
    with pytest.raises(PermissionError):
        mongo_service.insert_message(
            conversation_id=conv["id"],
            user_id="otro-usuario",
            role="user",
            content="Este debe fallar",
        )
