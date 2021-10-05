from forecast.simpful import *


class FpiSugeno(object):

    def __init__(self):
        # A simple fuzzy inference system for the tipping problem
        # Create a fuzzy system object
        self.FS = FuzzySystem(show_banner=False)

        # Poor (< 0.20), b. Fair (0.21 to 0.40), c. Moderate (0.41 to 0.60), d. Good (0.61 to 0.80), and e. Very Good (0.81 to 1.00)
        # Define fuzzy sets and linguistic variables
        S_1 = FuzzySet(points=[[0.00, 0.20], [0.00, 0.20]], term="poor")
        S_2 = FuzzySet(points=[[0.21, 0.40], [0.21, 0.40]], term="fair")
        S_3 = FuzzySet(points=[[0.41, 0.60], [0.41, 0.60]], term="moderate")
        S_4 = FuzzySet(points=[[0.61, 0.80], [0.61, 0.80]], term="good")
        S_5 = FuzzySet(points=[[0.81, 1.00], [0.81, 1.00]], term="excellent")
        self.FS.add_linguistic_variable("D", LinguisticVariable([S_1, S_2, S_3, S_4, S_5], concept="inden of agreement"))
        self.FS.add_linguistic_variable("E", LinguisticVariable([S_1, S_2, S_3, S_4, S_5], concept="Legates and McCabe’s"))
        self.FS.add_linguistic_variable("R", LinguisticVariable([S_1, S_2, S_3, S_4, S_5], concept="Berry and Mielke’s"))

        F_1 = FuzzySet(points=[[0.00, 0.20], [0.00, 0.20]], term="excellent")
        F_2 = FuzzySet(points=[[0.21, 0.40], [0.21, 0.40]], term="good")
        F_3 = FuzzySet(points=[[0.41, 0.60], [0.41, 0.60]], term="moderate")
        F_4 = FuzzySet(points=[[0.61, 0.80], [0.61, 0.80]], term="fair")
        F_5 = FuzzySet(points=[[0.81, 1.00], [0.81, 1.00]], term="poor")
        self.FS.add_linguistic_variable("U", LinguisticVariable([F_5, F_4, F_3, F_2, F_1], concept="Theil’s Inequality Coefficient"))


        # Define output crisp values
        self.FS.set_crisp_output_value("poor", 0.2)
        self.FS.set_crisp_output_value("fair", 0.4)
        self.FS.set_crisp_output_value("moderate", 0.6)
        self.FS.set_crisp_output_value("good", 0.8)
        self.FS.set_crisp_output_value("excellent", 1.0)

        # Define function for performance
        self.FS.set_output_function("Performance", "D+E+U+R")

        # Define fuzzy rules
        R11 = "IF (R IS poor) THEN (Performance IS poor)"
        R12 = "IF (U IS poor) THEN (Performance IS poor)"
        R13 = "IF (E IS poor) THEN (Performance IS poor)"
        R14 = "IF (D IS poor) THEN (Performance IS poor)"

        R21 = "IF (R IS fair) THEN (Performance IS fair)"
        R22 = "IF (U IS fair) THEN (Performance IS fair)"
        R23 = "IF (E IS fair) THEN (Performance IS fair)"
        R24 = "IF (D IS fair) THEN (Performance IS fair)"

        R31 = "IF (R IS moderate) THEN (Performance IS moderate)"
        R32 = "IF (U IS moderate) THEN (Performance IS moderate)"
        R33 = "IF (E IS moderate) THEN (Performance IS moderate)"
        R34 = "IF (D IS moderate) THEN (Performance IS moderate)"

        R41 = "IF (R IS good) THEN (Performance IS good)"
        R42 = "IF (U IS good) THEN (Performance IS good)"
        R43 = "IF (E IS good) THEN (Performance IS good)"
        R44 = "IF (D IS good) THEN (Performance IS good)"

        R51 = "IF (R IS excellent) THEN (Performance IS excellent)"
        R52 = "IF (U IS excellent) THEN (Performance IS excellent)"
        R53 = "IF (E IS excellent) THEN (Performance IS excellent)"
        R54 = "IF (D IS excellent) THEN (Performance IS excellent)"

        self.FS.add_rules([
            R11, R12, R13, R14,
            R21, R22, R23, R24,
            R31, R32, R33, R34,
            R41, R42, R43, R44,
            R51, R52, R53, R54
        ])



    def __call__(self, d, e, u, r):
        # Set antecedents values
        self.FS.set_variable("D", d)
        self.FS.set_variable("E", e)
        self.FS.set_variable("U", u)
        self.FS.set_variable("R", r)

        # Perform Sugeno inference and print output
        x = self.FS.Sugeno_inference(["Performance"])
        xx = None
        if isinstance(x, dict):
            if "Performance" in x:
                xx = x["Performance"]

        return xx




class FpiMamdani(object):

    def __init__(self):
        # A simple fuzzy inference system for the tipping problem
        # Create a fuzzy system object
        self.FS = FuzzySystem(show_banner=False)

        # Poor (< 0.20), b. Fair (0.21 to 0.40), c. Moderate (0.41 to 0.60), d. Good (0.61 to 0.80), and e. Very Good (0.81 to 1.00)
        # Define fuzzy sets and linguistic variables
        S_1 = FuzzySet(function=Triangular_MF(a=0.00, b=0.00, c=0.25), term="poor")
        S_2 = FuzzySet(function=Triangular_MF(a=0.00, b=0.25, c=0.50), term="fair")
        S_3 = FuzzySet(function=Triangular_MF(a=0.25, b=0.50, c=0.75), term="moderate")
        S_4 = FuzzySet(function=Triangular_MF(a=0.50, b=0.75, c=1.00), term="good")
        S_5 = FuzzySet(function=Triangular_MF(a=0.75, b=1.00, c=1.00), term="excellent")
        self.FS.add_linguistic_variable("D", LinguisticVariable([S_1, S_2, S_3, S_4, S_5], concept="inden of agreement"))
        self.FS.add_linguistic_variable("E", LinguisticVariable([S_1, S_2, S_3, S_4, S_5], concept="Legates and McCabe’s"))
        self.FS.add_linguistic_variable("R", LinguisticVariable([S_1, S_2, S_3, S_4, S_5], concept="Berry and Mielke’s"))

        F_1 = FuzzySet(function=Triangular_MF(a=0.00, b=0.00, c=0.25), term="excellent")
        F_2 = FuzzySet(function=Triangular_MF(a=0.00, b=0.25, c=0.50), term="good")
        F_3 = FuzzySet(function=Triangular_MF(a=0.25, b=0.50, c=0.75), term="moderate")
        F_4 = FuzzySet(function=Triangular_MF(a=0.50, b=0.75, c=1.00), term="fair")
        F_5 = FuzzySet(function=Triangular_MF(a=0.75, b=1.00, c=1.00), term="poor")
        self.FS.add_linguistic_variable("U", LinguisticVariable([F_5, F_4, F_3, F_2, F_1], concept="Theil’s Inequality Coefficient"))


        self.FS.add_linguistic_variable("Performance", LinguisticVariable([S_1, S_2, S_3, S_4, S_5], universe_of_discourse=[0,1.0]))

        # Define fuzzy rules
        R11 = "IF (R IS poor) THEN (Performance IS poor)"
        R12 = "IF (U IS poor) THEN (Performance IS poor)"
        R13 = "IF (E IS poor) THEN (Performance IS poor)"
        R14 = "IF (D IS poor) THEN (Performance IS poor)"

        R21 = "IF (R IS fair) THEN (Performance IS fair)"
        R22 = "IF (U IS fair) THEN (Performance IS fair)"
        R23 = "IF (E IS fair) THEN (Performance IS fair)"
        R24 = "IF (D IS fair) THEN (Performance IS fair)"

        R31 = "IF (R IS moderate) THEN (Performance IS moderate)"
        R32 = "IF (U IS moderate) THEN (Performance IS moderate)"
        R33 = "IF (E IS moderate) THEN (Performance IS moderate)"
        R34 = "IF (D IS moderate) THEN (Performance IS moderate)"

        R41 = "IF (R IS good) THEN (Performance IS good)"
        R42 = "IF (U IS good) THEN (Performance IS good)"
        R43 = "IF (E IS good) THEN (Performance IS good)"
        R44 = "IF (D IS good) THEN (Performance IS good)"

        R51 = "IF (R IS excellent) THEN (Performance IS excellent)"
        R52 = "IF (U IS excellent) THEN (Performance IS excellent)"
        R53 = "IF (E IS excellent) THEN (Performance IS excellent)"
        R54 = "IF (D IS excellent) THEN (Performance IS excellent)"

        self.FS.add_rules([
            R11, R12, R13, R14,
            R21, R22, R23, R24,
            R31, R32, R33, R34,
            R41, R42, R43, R44,
            R51, R52, R53, R54
        ])



    def __call__(self, d, e, u, r):
        # Set antecedents values
        self.FS.set_variable("D", d)
        self.FS.set_variable("E", e)
        self.FS.set_variable("U", u)
        self.FS.set_variable("R", r)

        y = self.FS.Mamdani_inference(["Performance"])
        yy = None
        if isinstance(y, dict):
            if "Performance" in y:
                yy = y["Performance"]

        return yy


if __name__ == '__main__':
    fpis = FpiSugeno()
    fpim = FpiMamdani()
    r = fpis(0.85, 0.45, 0.3, 0.5)
    print(r)
    r = fpim(0.85, 0.45, 0.3, 0.5)
    print(r)