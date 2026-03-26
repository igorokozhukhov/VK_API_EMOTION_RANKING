CITY_IDS_TATARSTAN = [
    73,
    204,
    99,
    1192,
    1187,
    11168,
]

WEIGHT_SOCIAL = 0.17
WEIGHT_ACTIVITY = 0.15
WEIGHT_SENTIMENT_MERGED = 0.25
WEIGHT_GROUPS = 0.15
WEIGHT_VISUAL = 0.13
WEIGHT_MUSIC = 0.15

PERCENTILE_LOW = 5.0
PERCENTILE_HIGH = 95.0

KEYWORDS_HOBBY = (
    "хобби", "спорт", "музык", "кино", "путешеств", "творчеств", "рисован",
    "фотограф", "танц", "игр", "книг", "чтен",
)
KEYWORDS_EDUCATION = (
    "университет", "школ", "образован", "курс", "лекц", "экзамен", "студен",
    "науч", "математ", "физик", "иностранн",
)
KEYWORDS_TOXIC = (
    "ненавист", "токсич", "хейт", "расизм", "оскорблен", "убийств",
)

MUSIC_POSITIVE_KEYWORDS = (
    "pop", "поп", "dance", "данс", "disco", "диско", "funk", "фанк",
    "reggae", "регги", "latin", "латин", "samba", "самба", "salsa",
    "happy", "love", "party", "summer", "sun", "joy", "fun", "dance",
    "smile", "dream", "beautiful", "good", "best", "light", "alive",
    "счастье", "любовь", "радость", "лето", "солнце", "танц", "мечта",
    "красив", "весел", "праздник", "свет", "улыбк", "позитив",
    "pharrell", "bruno mars", "maroon 5", "shakira", "abba",
    "dua lipa", "ed sheeran", "jason mraz", "bob marley",
    "макс корж", "егор крид", "мот", "ёлка", "artik",
)

MUSIC_NEGATIVE_KEYWORDS = (
    "death metal", "black metal", "doom", "grindcore", "deathcore",
    "depressive", "funeral", "dark ambient",
    "pain", "death", "hate", "cry", "tears", "sorrow", "dark", "blood",
    "kill", "hell", "grave", "suffer", "agony", "war", "rage", "broken",
    "боль", "смерть", "ненависть", "слёзы", "тоска", "печаль", "тьма",
    "кровь", "могил", "война", "страдан", "депресс", "одиноч", "плач",
    "burzum", "mayhem", "cannibal corpse", "slayer",
    "ghostemane", "bones", "suicideboys", "$uicideboy$",
)

MUSIC_GROUP_KEYWORDS = (
    "музык", "music", "rock", "рок", "рэп", "rap", "hip-hop", "хип-хоп",
    "pop", "поп", "jazz", "джаз", "electronic", "электрон", "metal", "метал",
    "punk", "панк", "indie", "инди", "soundtrack", "саундтрек", "classical",
    "классическ", "r&b", "ритм", "soul", "соул", "blues", "блюз",
    "techno", "техно", "house", "хаус", "trance", "транс", "edm",
    "spotify", "apple music", "яндекс музыка", "вк музыка", "boom",
    "playlist", "плейлист", "vinyl", "винил", "концерт", "concert",
    "festival", "фестиваль",
)
