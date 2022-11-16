import math
import numpy as np
import matplotlib.pyplot as plt

################################################################################################
# Сила Лоренца F = q ( E + v ^ B ) = m dv/dt
# В однородном магнитном поле B:
# траектория - окружность
# норма скорости не меняется, меняется только ее ориентация
# скорость и радиус связаны между собой v/r = B q/m
# время прохождения полного круга равно 2 PI / ( B q/m )
#
# В однородном электрическом поле E:
# траектория - парабола, либо прямая, в случае если скорость и E коллинеарны
# ускорение постоянно a = E q/m
# поэтому Δv = a Δt
################################################################################################

# SPEED OF LIGHT is 299792458 m/s
speed_of_light = 3e8
# MACHINE EPSILON
eps = np.finfo(float).eps


class Particle:
    def __init__(self, pos, vel, mass, charge):
        self.pos = pos
        self.vel = vel
        self.mass = mass
        self.charge = charge
        self.q_over_m = charge / mass


class Cyclotron:
    spacing = 0.01
    spacing_x0 = -spacing / 2

    # VOLTAGE
    voltage = 50000.0
    # ELECTRIC FIELD
    E = [voltage / spacing, 0.0, 0.0]
    # MAGNETIC FIELD
    B = [0.0, 0.0, 1.5]
    __b = np.linalg.norm(B)

    # array with jump times
    __T_j = []
    # jump counting flag
    __started_jumping = False
    # velocity we want to get as a result
    max_velocity = 0.045 * speed_of_light
    # D's radius
    max_R = None
    # maximum y coordinate
    max_y = None
    # particle holding time in the cyclotron
    __holding_time = 0.0
    # if true, the D's radius and max_velocity will be automatically calculated
    __auto_max_velocity = True
    # if true, the voltage sign will depend on parity of jumps number
    __auto_freq = True
    period = None
    final_velocity = None

    __num_points_of_last_circle = 400

    particle = None
    source_pos = None
    expected_period = None
    delta_t = None

    def set_spacing(self, spacing, x0):
        self.spacing = spacing
        self.spacing_x0 = x0
        self.E = [self.voltage / spacing, 0.0, 0.0]
        if self.expected_period is not None:
            self.delta_t = self.expected_period / (self.__num_points_of_last_circle * math.sqrt(0.05 / spacing))

    def set_voltage(self, vol):
        self.voltage = vol
        self.E = [vol / self.spacing, 0.0, 0.0]

    def set_b(self, b):
        self.__b = math.fabs(b)
        self.B = [0.0, 0.0, b]
        if self.particle is not None:
            self.set_particle(self.particle)

    def set_particle(self, particle):
        self.particle = particle
        self.source_pos = particle.pos
        if self.__b > eps:
            self.expected_period = np.fabs(2.0 * math.pi / (self.__b * self.particle.q_over_m))
            self.delta_t = self.expected_period / (self.__num_points_of_last_circle * math.sqrt(0.05 / self.spacing))
        else:
            self.expected_period = math.inf
            self.delta_t = 1e-10
        self.set_max_velocity(self.max_velocity)

    def set_max_velocity(self, vel):
        assert self.particle is not None, "Particle not set. Use set_particle first"
        self.__auto_max_velocity = True
        self.max_velocity = vel
        if self.__b > eps:
            self.max_R = vel / (self.particle.q_over_m * self.__b)
        else:
            self.max_R = math.inf

    def set_max_R(self, R):
        assert self.particle is not None, "Particle not set. Use set_particle first"
        self.__auto_max_velocity = True
        self.max_R = R
        if self.__b > eps:
            self.max_velocity = self.particle.q_over_m * self.__b * R
        else:
            self.max_velocity = math.inf

    def set_max_y(self, y):
        self.__auto_max_velocity = False
        self.max_y = y

    def set_period(self, period):
        self.__auto_freq = False
        self.period = period

    def enable_auto_freq(self):
        self.__auto_freq = True

    def reset(self):
        self.__started_jumping = False
        self.__T_j = []
        self.__holding_time = 0.0

    def get_result_period(self):
        if len(self.__T_j) == 0:
            return math.inf

        return 2.0 * (self.__T_j[-1] - self.__T_j[0]) / (len(self.__T_j) - 1)

    def is_inside_spacing(self, position):
        return self.spacing_x0 < position[0] < self.spacing_x0 + self.spacing

    def e_acceleration(self, q_over_m, position, time):
        if self.__auto_freq:
            sgn = 1 if len(self.__T_j) % 2 == 0 else -1
        else:
            sgn = 1 if ((time + self.period / 4) // (self.period / 2)) % 2 == 0 else -1
        return np.array(self.E) * (sgn * q_over_m)

    def m_acceleration(self, q_over_m, position, velocity, time):
        return np.cross(velocity, self.B) * q_over_m

    # returns the acceleration due to an electromagnetic field ( from Lorenz force )
    def acceleration(self, q_over_m, position, velocity, time):
        # stop after reaching max velocity or max y
        if self.__auto_max_velocity:
            if math.fabs(-velocity[0]) >= self.max_velocity and position[1] > self.source_pos[1]:
                return np.zeros(3)
        elif position[1] > self.max_y:
            return np.zeros(3)

        if not self.is_inside_spacing(position):
            if self.__started_jumping:
                print("  velocity is %1.4f the speed of light at %1.10f sec" % (
                    np.linalg.norm(velocity) / speed_of_light, time))
                self.__started_jumping = False
                self.__T_j.append(time)

            return self.m_acceleration(q_over_m, position, velocity, time)

        self.__started_jumping = True

        return self.e_acceleration(q_over_m, position, time) + self.m_acceleration(q_over_m, position, velocity, time)

    # Runge-Kutta method
    def rk4(self, max_time, delta_t):
        # initial conditions
        p = np.array(self.particle.pos)
        v = np.array(self.particle.vel)
        t = 0.0

        P = [p]
        V = [np.linalg.norm(v)]

        for _ in range(0, int(max_time // delta_t)):
            t += delta_t

            p1 = p
            v1 = v
            a1 = delta_t * self.acceleration(self.particle.q_over_m, p1, v1, t)
            v1 = delta_t * v1

            p2 = p + (v1 * 0.5)
            v2 = v + (a1 * 0.5)
            a2 = delta_t * self.acceleration(self.particle.q_over_m, p2, v2, t)
            v2 *= delta_t

            p3 = p + (v2 * 0.5)
            v3 = v + (a2 * 0.5)
            a3 = delta_t * self.acceleration(self.particle.q_over_m, p3, v3, t)
            v3 *= delta_t

            p4 = p + v3
            v4 = v + a3
            a4 = delta_t * self.acceleration(self.particle.q_over_m, p4, v4, t)
            v4 *= delta_t

            dv = a1 + 2.0 * (a2 + a3) + a4
            v = v + dv / 6.0

            dp = v1 + 2.0 * (v2 + v3) + v4
            p = p + dp / 6.0

            # if the acceleration is over, track the particle in the direction of the final velocity
            if np.allclose(dv, [0., 0., 0.]):
                self.__holding_time = t
                vn = np.linalg.norm(v)
                P = np.concatenate((P, np.linspace(p, p + v * (2 * np.linalg.norm(p) / vn), 100)), axis=0)
                V = np.concatenate((V, np.full(100, vn)), axis=0)
                return P, V

            P.append(p)
            V.append(np.linalg.norm(v))

        self.__holding_time = max_time
        return P, V

    def run(self, max_time, draw_plots=True):
        assert self.particle is not None, "Particle not set. Use set_particle first"

        self.reset()
        self.print_init()

        P, V = self.rk4(max_time, self.delta_t)
        self.final_velocity = V[-1]

        self.print_output()
        if draw_plots:
            self.draw_plots(P, V)

        return P, V

    def print_init(self):
        print("Particle:\n  mass=" + str(self.particle.mass) + " kg\n" +
              "  charge=" + str(self.particle.charge) + " C\n" +
              "  v_i=" + str(np.linalg.norm(self.particle.vel)) + " m/s\n" +
              "Uniform B=" + str(np.sign(self.B[2]) * np.linalg.norm(self.B)) + " Tesla\n" +
              "Uniform E=" + str(np.sign(self.E[0]) * np.linalg.norm(self.E)) + " V/m\n" +
              "Spacing=" + str(self.spacing) + " m\n")
        print("expected period is %1.20f seconds" % self.expected_period)
        print("expected frequency is %1.4f MHz" % (1.0 / self.expected_period / 1e6))
        if self.__auto_max_velocity:
            print("D`s radius must be more than", self.max_R, "m")
            print("expected maximum velocity is %f the speed of light" % (self.max_velocity / speed_of_light))
        print("initial position", self.particle.pos)
        print("initial velocity %1.4f the speed of light" % (np.linalg.norm(self.particle.vel) / speed_of_light))

    def print_output(self):
        print("final velocity %1.4f the speed of light" % (np.linalg.norm(self.final_velocity) / speed_of_light))
        print("number of jumps between D's is", len(self.__T_j))
        print("acceleration time is %1.20f seconds" % self.__holding_time)
        print("resulting period is %1.20f seconds" % self.get_result_period())
        print("resulting frequency is %1.4f MHz" % (1.0 / self.get_result_period() / 1e6))
        if not self.__auto_max_velocity:
            print("D`s radius must be more than", np.linalg.norm(self.final_velocity) / (self.particle.q_over_m * self.__b), "m")

    def draw_plots(self, P, V):
        # MATPLOTLIB PARAMS
        ax1 = plt.figure().add_subplot(1, 1, 1)
        ax2 = plt.figure().add_subplot(1, 1, 1)

        ax1.scatter(self.source_pos[0], self.source_pos[1], color='blue')

        # save the positions when in the spacing in a separate array
        # so that w can change the color to red
        xc = []
        yc = []
        x = []
        y = []
        for p in P:
            if not self.is_inside_spacing(p):
                #                  inside the D's
                if len(xc):
                    ax1.plot(xc, yc, color='red', linewidth=0.95)
                    xc = []
                    yc = []
                x.append(p[0])
                y.append(p[1])
            else:
                #                  inside the spacing
                if len(xc):
                    ax1.plot(x, y, color='green', linewidth=0.95)
                    x = []
                    y = []
                xc.append(p[0])
                yc.append(p[1])

        if len(xc):
            ax1.plot(xc, yc, color='red', linewidth=0.95)
        if len(x):
            ax1.plot(x, y, color='green', linewidth=0.95)

        ax1.axis('equal')
        ax1.set_title("Trajectory of the particle - Cyclotron")
        ax1.set_xlabel("Dimension-X (m)")
        ax1.set_ylabel("Dimension-Y (m)")

        t = np.linspace(0, len(V) * self.delta_t, len(V))
        ax2.plot(t, V)
        ax2.set_title("Speed of the particle as a function of time")
        ax2.set_xlabel("Time (sec)")
        ax2.set_ylabel("Speed (m/s)")

        plt.show()


def main():
    max_time = 1e-5
    d = 0.05
    voltage = 50000.0
    b = 1.5
    max_velocity = 0.05 * speed_of_light
    proton = Particle([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.67E-27, +1.60E-19)

    cyclotron = Cyclotron()
    cyclotron.set_particle(proton)
    cyclotron.set_voltage(voltage)
    cyclotron.set_b(b)
    cyclotron.set_max_velocity(max_velocity)
    cyclotron.set_spacing(d, -d / 2)

    cyclotron.run(max_time)


if __name__ == '__main__':
    main()
