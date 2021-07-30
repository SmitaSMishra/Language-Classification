class Node:
    __slots__ = "attribute", "dutchCount", "italianCount", "left", "right", "prediction"

    def __init__(self,attribute,dutchCount,italianCount, prediction = None):
        self.attribute = attribute
        self.dutchCount = dutchCount
        self.italianCount = italianCount
        self.left = None
        self.right = None
        self.prediction = prediction

    def __str__(self):
        return f' ATTR: {str(self.attribute)} DU: {str(self.dutchCount)} IT:{str(self.italianCount)} PREDICTION:{str(self.prediction)} LEFT:{str(self.left)} RIGHT:{str(self.right)}'

