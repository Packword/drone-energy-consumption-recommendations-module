import math
import scipy
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Point:
    def __init__(self, x, y, z, isKeyPoint=False, hover_time=0):
        # Координаты точки в трехмерном пространстве
        self.coordinates = np.array([x, y, z])

        # Флаг, указывающий, является ли эта точка ключевой (True) или нет (False)
        self.isKeyPoint = isKeyPoint

        # Энергия, необходимая для перемещения к следующей точке в траектории
        self.energy_to_next_point = 0

        # Энергия, необходимая для возвращения к базовой точке (обычно база - это точка старта)
        self.energy_to_base = 0

        # Энергия, необходимая для перемещения к следующей ключевой точке в траектории
        self.next_key_point_energy = 0

        # Индекс следующей ключевой точки в массиве точек
        self.next_key_point_index = 0

        # Время, на которое дрон зависает в этой точке
        self.hover_time = hover_time

        # Энергия, необходимая для парения на месте (без движения) определённое время hover_time
        self.energy_for_hover = 0

        # Критическое значение энергии до следующей ключевой точки (расчетное значение для контроля энергии)
        self.critical_value_next_key_point = 0

        # Критическое значение энергии до следующей точки в общей траектории (расчетное значение для контроля энергии)
        self.critical_value_next_point = 0


    def __str__(self):
      return f'Coordinates: {self.coordinates}, isKeyPoint: {self.isKeyPoint}, ' \
             f'energy_to_next_point: {self.energy_to_next_point}, energy_to_base: {self.energy_to_base}, ' \
             f'next_key_point_energy: {self.next_key_point_energy}, next_key_point_index: {self.next_key_point_index}, ' \
             f'hover_time: {self.hover_time}, energy_for_hover: {self.energy_for_hover}, critical_value_next_key_point: {self.critical_value_next_key_point}'

class DroneStatus(Enum):
  OK = 1
  RETURN = 2
  TAKEOVER = 3

class EnergyConsumptionModel:
  def calculate_energy_between_points(self, start_point, end_point, va_horizontal, va_vertical):
    pass
  def calculate_energy_for_hover(self, time):
    pass

class DroneConsumptionModelFor3DReconstruction:
  def __init__(self, energy_model, trajectory_builder = lambda x, y: [x,y]):
    if not issubclass(type(energy_model), EnergyConsumptionModel):
      raise Exception("Yours energy model must inheritaned EnergyConsumptionModel class")
    if not callable(trajectory_builder):
      raise ValueError("trajectory_builder must be a callable function")
    self.energy_consumption_model = energy_model
    self.trajectory_builder = trajectory_builder

  # Поиск оптимальной скорости. На вход идёт функция, которая принимает va и *args, далее вызывает нужный нам метод и возрващает результат, на который мы ориентируемся при миниммизации
  def find_optimal_speed(self, objective_function, initial_va, args):
    if not callable(objective_function):
      raise ValueError("objective_function must be a callable function")
    if not isinstance(initial_va, (int, float)):
      raise ValueError("initial_va must be a numeric value")
    if initial_va <= 0:
      raise ValueError("initial_va must be greater than 0")

    return scipy.optimize.minimize(objective_function, initial_va, args=args, bounds=[(0.1, self.energy_consumption_model.max_stable_speed)]).x[0]

  # Расчёт максимальной скорости, с которой можно добраться от a до b и при этом не кончится энергия. За рамки максимальной стабильной скорости не выходим.
  # Тут используется бинарный поиск.
  def calculate_max_speed(self, start_point, end_point, remaining_energy, tolerance=0.01):
    low_speed = 0.1
    high_speed = self.energy_consumption_model.max_stable_speed
    optimal_speed = 0
    points_to_base = self.trajectory_builder(start_point, end_point)
    while low_speed <= high_speed:
        mid_speed = (low_speed + high_speed) / 2
        energy_consumption = 0
        for j in range(len(points_to_base) - 1):
          energy_consumption += self.energy_consumption_model.calculate_energy_between_points(points_to_base[j], points_to_base[j + 1], mid_speed, mid_speed)

        if energy_consumption * 1.2 <= remaining_energy:
            optimal_speed = mid_speed
            low_speed = mid_speed + tolerance
        else:
            high_speed = mid_speed - tolerance

    return optimal_speed

  # Расстановка проежуточных точек на траектории
  def add_intermediate_points(self, points, range_between_points):
    if not isinstance(points, list):
      raise ValueError("points must be a list of Point instances")
    if not all(isinstance(point, Point) for point in points):
      raise ValueError("All elements in points must be instances of Point class")
    if not isinstance(range_between_points, (int, float)):
      raise ValueError("range_between_points must be a numeric value")
    if range_between_points <= 0:
      raise ValueError("range_between_points must be greater than 0")
    result = [points[0]]  # Начинаем с первой точки
    for i in range(1, len(points)):
        distance = np.linalg.norm(points[i].coordinates - points[i-1].coordinates)  # Вычисляем расстояние между текущей и предыдущей точками
        if distance > range_between_points:
            num_points = int(distance // range_between_points)  # Определяем количество промежуточных точек
            interval = distance / (num_points + 1)  # Вычисляем расстояние между промежуточными точками
            direction = (points[i].coordinates - points[i-1].coordinates) / distance  # Направление между точками
            for j in range(num_points):
                intermediate_coordinates = points[i-1].coordinates + direction * interval * (j + 1)  # Вычисляем координаты промежуточной точки
                intermediate_point = Point(*intermediate_coordinates)
                result.append(intermediate_point)
        result.append(points[i])  # Добавляем текущую точку
    return result

  # Расчёт суммарно затраченной энергии, разбиение траектории на участки.
  # Тут используется система "НА ВСЯКИЙ СЛУЧАЙ". На всякий случай, когда считаем сколько энергии надо,
  # то умножаем это на 1.3, чтобы не было такого, что пока мы летим внезапно кончится энергия и мы упадём
  def calculate_returning_points_for_trajectory(self, points, va_horizontal, va_vertical, base_point):
    if not isinstance(base_point, Point):
      raise ValueError("base_point must be instance of Point class")
    if not isinstance(va_horizontal, (int, float)):
      raise ValueError("va_horizontal must be a numeric value")
    if not isinstance(va_vertical, (int, float)):
      raise ValueError("va_vertical must be a numeric value")
    if va_horizontal <= 0:
      raise ValueError("va_horizontal must be greater than 0")
    if va_vertical <= 0:
      raise ValueError("va_vertical must be greater than 0")
    if not isinstance(points, list):
      raise ValueError("points must be a list of Point instances")
    if not all(isinstance(point, Point) for point in points):
      raise ValueError("All elements in points must be instances of Point class")


    returning_points = []
    remaining_energy = self.energy_consumption_model.battery_capacity
    total_energy_wasted = 0
    last_key_point_index = 0

    self.calculate_energy_for_points(points, va_horizontal, va_vertical, base_point)

    remaining_energy -= points[0].energy_to_base
    total_energy_wasted += points[0].energy_to_base

    for i in range(len(points) - 1):
      if points[i].isKeyPoint:
        remaining_energy -= points[i].energy_for_hover
        total_energy_wasted += points[i].energy_for_hover
        if remaining_energy - points[i].critical_value_next_key_point * 1.3 <= 0:
          returning_points.append(points[i])
          total_energy_wasted += (points[points[i].next_key_point_index].energy_to_base + points[i].energy_to_base)
          remaining_energy = self.energy_consumption_model.battery_capacity - points[points[i].next_key_point_index].energy_to_base
        else:
          remaining_energy -= points[i].next_key_point_energy
          total_energy_wasted += points[i].next_key_point_energy
        last_key_point_index = i

    remaining_energy -= points[len(points) - 1].energy_for_hover
    total_energy_wasted += points[len(points) - 1].energy_for_hover
    remaining_energy -= points[len(points) - 1].energy_to_base
    total_energy_wasted += points[len(points) - 1].energy_to_base
    return returning_points, remaining_energy, total_energy_wasted

  # Контроль актуального статуса при полёте, возвращает команду и скорость для её выполнения, если это Return, в остальных случаях стандартная скорость.
  def current_status_control(self, point, base_point, remaining_energy):
    if not isinstance(point, Point):
      raise ValueError("base_point must be instance of Point class")
    if not isinstance(point, Point):
      raise ValueError("point must be instance of Point class")
    if not isinstance(remaining_energy, (int, float)):
      raise ValueError("remaining_energy must be a numeric value")
    if remaining_energy < 0:
      raise ValueError("remaining_energy must be greater or equal than 0")

    if point.energy_to_base * 1.2 > remaining_energy:
      return DroneStatus.TAKEOVER, 0

    if point.critical_value_next_point >= remaining_energy or point.critical_value_next_key_point >= remaining_energy:
      return DroneStatus.RETURN, self.calculate_max_speed(point, base_point, remaining_energy)

    return DroneStatus.OK, 0

  # Маркировка точек
  def calculate_energy_for_points(self, points, va_horizontal, va_vertical, base_point):
    if not isinstance(base_point, Point):
      raise ValueError("base_point must be instance of Point class")
    if not isinstance(points, list):
      raise ValueError("points must be a list of Point instances")
    if not all(isinstance(point, Point) for point in points):
      raise ValueError("All elements in points must be instances of Point class")
    if not isinstance(va_horizontal, (int, float)):
      raise ValueError("va_horizontal must be a numeric value")
    if not isinstance(va_vertical, (int, float)):
      raise ValueError("va_vertical must be a numeric value")
    if va_horizontal <= 0:
      raise ValueError("va_horizontal must be greater than 0")
    if va_vertical <= 0:
      raise ValueError("va_vertical must be greater than 0")
    next_key_point_index = len(points) - 1
    next_key_point_energy = 0

    # Рассчитываем энергию для каждой точки, начиная с конца
    for i in range(len(points) - 1, 0, -1):
        points[i].next_key_point_index = next_key_point_index

        if points[i].isKeyPoint:
            next_key_point_index = i
            points[i].energy_for_hover = self.energy_consumption_model.calculate_energy_for_hover(points[i].hover_time)
            next_key_point_energy = 0

        # Рассчитываем энергию до базовой точки и до следующей ключевой точки
        points_to_base = self.trajectory_builder(points[i], base_point)
        for j in range(len(points_to_base) - 1):
          points[i].energy_to_base += self.energy_consumption_model.calculate_energy_between_points(points_to_base[j], points_to_base[j + 1], va_horizontal, va_vertical)

        points[i - 1].energy_to_next_point += self.energy_consumption_model.calculate_energy_between_points(points[i - 1], points[i], va_horizontal, va_vertical)

        # Обновляем энергию до следующей ключевой точки
        next_key_point_energy += points[i - 1].energy_to_next_point
        points[i - 1].next_key_point_energy = next_key_point_energy

        # Рассчитываем критические значения энергии
        points[i - 1].critical_value_next_key_point = (next_key_point_energy +
                                                       points[points[i - 1].next_key_point_index].energy_to_base +
                                                       points[points[i - 1].next_key_point_index].energy_for_hover) * 1.3
        points[i - 1].critical_value_next_point = (points[i - 1].energy_to_next_point +
                                                   points[i].energy_to_base) * 1.5

    # Рассчитываем энергию для первой точки
    points[0].next_key_point_index = next_key_point_index
    points[0].energy_for_hover = self.energy_consumption_model.calculate_energy_for_hover(points[0].hover_time)
    points_to_base = self.trajectory_builder(points[0], base_point)
    for j in range(len(points_to_base) - 1):
      points[0].energy_to_base += self.energy_consumption_model.calculate_energy_between_points(points_to_base[j], points_to_base[j + 1], va_horizontal, va_vertical)

class EnergyConsumptionModelR2(EnergyConsumptionModel):
  def __init__(self, drag_coef, p, M, nu, r, n, capacity, max_stable_speed, projectred_area):
    if not all(isinstance(param, (int, float)) for param in (drag_coef, p, M, nu, r, n, capacity, max_stable_speed, projectred_area)):
      raise TypeError("All parameters must be numeric values")
    if not all(param > 0 for param in (drag_coef, p, M, nu, r, n, capacity, max_stable_speed, projectred_area)):
        raise ValueError("All parameters must be greater than 0")
    self.drag_coef = drag_coef
    self.air_density = p
    self.M = M
    self.battery_transfer_coef = nu
    self.radius_blades = r
    self.number_blades = n
    self.battery_capacity = capacity
    self.max_stable_speed = max_stable_speed
    self.projectred_area = projectred_area

  # Энергия на метр
  def EPM(self, P, va):
    return P/va;

  # Мощность горизонтального полёта
  def P_Forward(self, va):
    alpha = self._calculate_alpha(va)
    vi = self._calculate_vi(alpha, va)
    return (self._T_Forward(va) * (va * math.sin(alpha) + vi)) / self.battery_transfer_coef

  # Мощность зависания
  def P_Hover(self):
    return ((scipy.constants.g * self.M) ** 1.5) / math.sqrt(2 * self.number_blades * self.air_density * 2 * scipy.pi * (self.radius_blades ** 2))

  # Мощность взлёта и посадки(одинаковые)
  def P_Takeover(self, va):
    vi = self._calculate_vi(0, va)
    return (self._T_Forward(va) * vi) / self.battery_transfer_coef

  def P_Landing(self, va):
    vi = self._calculate_vi(0, va)
    return (self._T_Forward(va) * vi) / self.battery_transfer_coef

  # Энергия между двумя точками
  def calculate_energy_between_points(self, start_point, end_point, va_horizontal, va_vertical):
    if not isinstance(va_horizontal, (int, float)):
      raise ValueError("va_horizontal must be a numeric value")
    if not isinstance(va_vertical, (int, float)):
      raise ValueError("va_vertical must be a numeric value")
    if va_horizontal <= 0:
      raise ValueError("va_horizontal must be greater than 0")
    if va_vertical <= 0:
      raise ValueError("va_vertical must be greater than 0")
    dx = end_point.coordinates[0] - start_point.coordinates[0]
    dy = end_point.coordinates[1] - start_point.coordinates[1]
    dz = end_point.coordinates[2] - start_point.coordinates[2]

    P = 0.0
    va = 0
    if dx == 0 and dz == 0 and dy > 0:
        va = va_vertical
        P = self.P_Takeover(va_vertical)
    elif dx == 0 and dz == 0 and dy < 0:
      va = va_vertical
      P = self.P_Landing(va_vertical)
    else:
        va = va_horizontal
        P = self.P_Forward(va_horizontal)
    return self.EPM(P, va) * np.linalg.norm(start_point.coordinates - end_point.coordinates)

  # Энергия для зависания на время t
  def calculate_energy_for_hover(self, time):
    if not isinstance(time, (int, float)):
      raise ValueError("time must be a numeric value")
    if time <= 0:
      raise ValueError("time must be greater than 0")
    return self.P_Hover() * time

  def _calculate_alpha(self, va):
    return np.arctan((0.5 * self.air_density * self.drag_coef * self.projectred_area * (va ** 2)) / (self.M * scipy.constants.g))

  def _T_Forward(self, va):
    return 0.5 * self.air_density * self.drag_coef * self.projectred_area * (va ** 2) + scipy.constants.g * self.M

  def _calculate_vi(self, alpha, va):
    vi_prev = 0;
    vi_next_cur = self._vi_next(0, alpha, va)
    while(abs(vi_prev - vi_next_cur) > 0.00001):
      vi_prev = vi_next_cur
      vi_next_cur = self._vi_next(vi_prev, alpha, va)
    return vi_next_cur

  def _vi_next(self, vi, alpha, va):
    return (scipy.constants.g * self.M) / (2 * self.number_blades * scipy.pi * (self.radius_blades ** 2) * self.air_density * math.sqrt((va * math.cos(alpha)) ** 2 + (va * math.sin(alpha) + vi) ** 2))

def visualize_points(points, returning_points = []):
    x = [point.coordinates[0] for point in points]
    y = [point.coordinates[1] for point in points]
    z = [point.coordinates[2] for point in points]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    normal_points = [point for point in points if not point.isKeyPoint]
    ax.scatter([point.coordinates[0] for point in normal_points],
               [point.coordinates[2] for point in normal_points],
               [point.coordinates[1] for point in normal_points],
               color='b', marker='o', label='Normal Points', zorder=1)

    key_points = [point for point in points if point.isKeyPoint and point not in returning_points]
    ax.scatter([point.coordinates[0] for point in key_points],
               [point.coordinates[2] for point in key_points],
               [point.coordinates[1] for point in key_points],
               color='r', marker='o', label='Key Points', zorder=2)

    ax.scatter([point.coordinates[0] for point in returning_points],
               [point.coordinates[2] for point in returning_points],
               [point.coordinates[1] for point in returning_points],
               color='g', marker='o', label='Returning points', depthshade=False, zorder=3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.legend()

    plt.show()

base_point = Point(0,0,0)

points = []
xf = 5  # Фиксированное значение x
y_step = 0.5  # Шаг для координаты y
z = 1

def objective_function_forward(va, *args):
    drag_coef, p, M, nu, r, n, capacity, speed, pa = args
    model = EnergyConsumptionModelR2(drag_coef, p, M, nu, r, n, capacity, speed, pa)
    result = model.P_Forward(va)
    return model.EPM(result, va)

def objective_function_takeover(va, *args):
    drag_coef, p, M, nu, r, n, capacity, speed, pa = args
    model = EnergyConsumptionModelR2(drag_coef, p, M, nu, r, n, capacity, speed, pa)
    result = model.P_Takeover(va)
    return model.EPM(result, va)

for i in range(1, 12):
  points.append(Point(-xf, y_step * 2 * i, z, True, 3))
  points.append(Point(xf, y_step * 2 * i, z, True, 3))
  points.append(Point(xf, y_step * 2 * i + y_step, z, True, 3))
  points.append(Point(-xf, y_step * 2 * i + y_step, z, True, 3))

visualize_points(points)

model = EnergyConsumptionModelR2(1, 1.2, 0.1, 0.7, 0.03, 4, 500, 5, 0.05)
recomendationsModel = DroneConsumptionModelFor3DReconstruction(model)

points = recomendationsModel.add_intermediate_points(points, 0.5)

speed_for_forward = recomendationsModel.find_optimal_speed(objective_function_forward, 0.1, args = (1, 1.2, 0.1, 0.7, 0.03, 4, 3000, 5, 0.05))
speed_for_takeover = recomendationsModel.find_optimal_speed(objective_function_takeover, 0.1, args = (1, 1.2, 0.1, 0.7, 0.03, 4, 3000, 5, 0.05))

print(speed_for_forward, speed_for_takeover)
print(model.EPM(model.P_Forward(speed_for_forward), speed_for_forward), model.EPM(model.P_Takeover(speed_for_takeover), speed_for_takeover))
print(model.P_Hover(), model.P_Forward(speed_for_forward), model.P_Takeover(speed_for_takeover))

visualize_points(points)

returning_points, remaining_energy, total_wasted = recomendationsModel.calculate_returning_points_for_trajectory(points, 6, 6, base_point)
print(total_wasted)
print(returning_points[0])

visualize_points(points, returning_points)
