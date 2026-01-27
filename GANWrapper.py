from GANGenerate import generateNewLevel
import numpy as np

class GANWrapper:
    def generate(self, vector_1, vector_2):
        try:
            values_1 = vector_1.tolist()
            values_2 = vector_2.tolist()
            level = generateNewLevel(values_1, values_2)
            return level.astype(np.int32)
        except Exception as e:
            return self._empty_level()
        
    def _empty_level(self):
        return np.zeros((14, 32), dtype=np.int32)