
import numpy as np


class InverseKinematics:
    def __init__(self, upper_arm_length=0.5, forearm_length=0.4):
        self.L1 = upper_arm_length
        self.L2 = forearm_length

    def calculate(self, x, y):
        """
        Calculate shoulder and elbow angles
        for a 2-link planar robotic arm.
        """

        distance_squared = x**2 + y**2

        cos_q2 = (
            distance_squared - self.L1**2 - self.L2**2
        ) / (2 * self.L1 * self.L2)

        # Target is outside the reachable workspace
        if cos_q2 < -1 or cos_q2 > 1:
            raise ValueError("Target is outside the reachable workspace")

        q2 = np.arccos(cos_q2)

        q1 = (
            np.arctan2(y, x)
            - np.arctan2(
                self.L2 * np.sin(q2),
                self.L1 + self.L2 * np.cos(q2)
            )
        )

        return q1, q2
