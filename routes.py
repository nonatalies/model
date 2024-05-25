import geopy.distance

# Данные достопримечательностей: {название: (широта, долгота)}
landmarks = {
    'Блиновский пассаж': (56.328234, 43.988602),  # Пример координат (широта, долгота)
    'Государственный банк': (56.320168, 43.998923),
    'Кафедральный собор во имя Святого Благоверного Князя Александра Невского': (56.333562, 43.971162),
    'Нижегородская соборная мечеть': (56.323235, 44.037701),
    'Нижегородская ярмарка': (56.328154, 43.961128),
    'Нижегородский кремль': (56.328624, 44.002842),
    'Пакгаузы': (56.335288, 43.973781),
    'Памятник гражданину Минину и князю Пожарскому': (56.329855, 43.996872),
    'Памятник Максиму Горькому': (56.313484, 43.990670),
    'Печерский Вознесенский монастырь': (56.323059, 44.049772),
    'Речной вокзал': (56.329622, 43.987858),
    'Усадьба С. М. Рукавишникова': (56.329295, 44.016300),
    'Храм в честь Собора Пресвятой Богородицы': (56.327287, 43.984909),
    'Чкаловская лестница': (56.330871, 44.009554)
}

def calculate_distance(coord1, coord2):
    return geopy.distance.distance(coord1, coord2).km

def find_nearest_landmark(current_landmark, landmarks, visited):
    nearest_landmark = None
    min_distance = float('inf')

    for landmark, coord in landmarks.items():
        if landmark not in visited:
            distance = calculate_distance(landmarks[current_landmark], coord)
            if distance < min_distance:
                min_distance = distance
                nearest_landmark = landmark

    return nearest_landmark, min_distance

def create_route(start_landmark, max_stops):
    visited = [start_landmark]
    current_landmark = start_landmark
    total_distance = 0

    while len(visited) < max_stops and len(visited) < len(landmarks):
        next_landmark, distance = find_nearest_landmark(current_landmark, landmarks, visited)
        visited.append(next_landmark)
        total_distance += distance
        current_landmark = next_landmark

    # Средняя скорость человека
    average_speed_kmh = 4
    total_time_hours = total_distance / average_speed_kmh

    # Преобразование времени в часы и минуты
    hours = int(total_time_hours)
    minutes = round((total_time_hours - hours) * 60)

    return visited, total_distance, hours, minutes

