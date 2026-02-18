# initail: generate description, semanticless or object listing sequence

class DescriptionGenerator:

    def generate(self, objects):
        """
        Convert structured geometry into natural spatial language.
        """

        if len(objects) == 0:
            return "No significant objects detected."

        # Sort by real distance (nearest first)
        objects = sorted(objects, key=lambda x: x["distance_m"])

        sentences = []

        for obj in objects:
            sentence = (
                f"There is an object located to the {obj['direction']} "
                f"at approximately {obj['distance_m']:.2f} meters"
                f"({obj['distance_label']})."
            )
            sentences.append(sentence)

        return " ".join(sentences)
