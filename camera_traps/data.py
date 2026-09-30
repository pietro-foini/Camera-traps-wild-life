from enum import StrEnum


class ClassifierClasses(StrEnum):
    """Enum for wildlife classifier classes"""

    human = "human"
    vehicle = "vehicle"
    bird = "bird"
    cow = "cow"
    fox = "fox"
    hare = "hare"
    dog = "dog"
    cat = "cat"
    none_of_the_above = "None_of_the_above"
    weasel = "weasel"
    bear = "bear"
    deer = "deer"
    badger = "badger"
    squirrel = "squirrel"
    wolf = "wolf"
    boar = "boar"
