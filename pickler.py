import datetime

import pickle


def load_all() -> dict:
    with open("pickles.pkl", 'rb+') as fp:
        try:
            return pickle.load(fp)
        except Exception as e:
            return {}


def save_all(data: dict):
    with open("pickles.pkl", 'wb+') as fp:
        pickle.dump(data, fp)


def get_full_user_data(user_id):
    return load_all().get(user_id, {})


def save_user_data(user_id, user_data):
    data = load_all()
    data[user_id] = user_data
    save_all(data)
    return data


def add_user_mood(user_id: int, mood: int):
    data = get_full_user_data(user_id)
    mood_data = data.get("_mood", [])
    mood_data.append((datetime.datetime.now(), mood))
    data["_mood"] = mood_data
    save_user_data(user_id, data)


def get_user_mood(user_id: int):
    data = get_full_user_data(user_id)
    mood_data = data.get("_mood", [])
    return mood_data


def set_custom_data(user_id, key, custom_data):
    data = get_full_user_data(user_id)
    data[key] = custom_data
    save_user_data(user_id, data)


def get_custom_data(user_id, key):
    data = get_full_user_data(user_id)
    return data.get(key, {})


def purge():
    save_all({})


def clean_user_data(user_id: int):
    save_user_data(user_id, {})
