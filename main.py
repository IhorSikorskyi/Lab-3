from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from deap import base, creator, tools, algorithms
import random

# Завантаження і підготовка даних
digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Фітнес-функція: створення та оцінка моделі
def evaluate_nn(individual):
    try:
        num_layers = len(individual) // 2

        # Витягуємо кількість нейронів і активацій
        neurons = individual[:num_layers]
        activations = individual[num_layers:]

        # Список дозволених активацій
        valid_activations = {'relu', 'sigmoid', 'tanh', 'softmax'}

        # Додатково перевіряємо форму особини
        if not len(neurons) == len(activations):
            raise ValueError(f"Кількість нейронів ({len(neurons)}) "
                             f"і активацій ({len(activations)}) не збігається.")

        # Перевірка: нейрони мають бути тільки додатніми цілими числами
        if not all(isinstance(n, int) and n > 0 for n in neurons):
            raise ValueError(f"Некоректні значення нейронів: {neurons}. "
                             "Всі значення мають бути додатними цілими числами.")

        # Перевірка: активації мають бути валідними рядковими значеннями
        if not all(isinstance(act, str) and act in valid_activations for act in activations):
            raise ValueError(f"Некоректні активації: {activations}. "
                             f"Допустимі значення: {valid_activations}.")


        # Створення моделі
        model = Sequential()

        # Додавання наступних шарів
        for i in range(0, num_layers):
            model.add(Dense(units=neurons[i], activation=activations[i]))

        # Вихідний шар
        model.add(Dense(units=10, activation='softmax'))

        # Компіляція та навчання
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0)

        # Оцінка моделі
        _, accuracy = model.evaluate(X_test, y_test, verbose=0)
        return accuracy,  # Повертаємо точність як значення фітнес-функції
    except Exception as e:
        raise RuntimeError(f"Помилка під час виконання функції evaluate_nn: {repr(e)}")


# Генетичний алгоритм
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Максимізація точності
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Генерація особин

# Діапазони значень гіперпараметрів
NEURONS_MIN, NEURONS_MAX = 8, 128
LAYERSS_MIN, LAYERSS_MAX = 1, 5
valid_activations = ['relu', 'sigmoid', 'tanh', 'softmax']

# Генерація параметрів для особини
toolbox.register("neurons", random.randint, NEURONS_MIN, NEURONS_MAX)
toolbox.register("num_layers", random.randint, LAYERSS_MIN, LAYERSS_MAX)
toolbox.register("activation", random.choice, valid_activations)

def create_individual():
    num_layers = random.randint(LAYERSS_MIN, LAYERSS_MAX)
    # Генерація тільки валідних значень
    neurons = [random.randint(NEURONS_MIN, NEURONS_MAX) for _ in range(num_layers)]
    activations = [random.choice(valid_activations) for _ in range(num_layers)]
    individual = neurons + activations
    return individual


toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Реєстрація операторів ГА

def fix_individual(individual):
    """Коригує особину для уникнення змішаних значень."""
    num_layers = len(individual) // 2
    neurons = individual[:num_layers]
    activations = individual[num_layers:]

    # Фіксуємо кількість нейронів
    for i in range(num_layers):
        if not isinstance(neurons[i], int) or neurons[i] < NEURONS_MIN or neurons[i] > NEURONS_MAX:
            neurons[i] = random.randint(NEURONS_MIN, NEURONS_MAX)

    # Фіксуємо активації
    for i in range(len(activations)):
        if not isinstance(activations[i], str) or activations[i] not in valid_activations:
            activations[i] = random.choice(valid_activations)

    # Гарантуємо коректність структури особини
    individual[:] = neurons + activations


# Виправлення помилки:
def custom_mutate(individual, indpb):
    """Мутує особину і виправляє проблеми після мутації."""
    num_layers = len(individual) // 2

    # Мутація кількості нейронів
    for i in range(num_layers):
        if random.random() < indpb:
            individual[i] = random.randint(NEURONS_MIN, NEURONS_MAX)

    # Мутація функції активації
    for i in range(num_layers, len(individual)):
        if random.random() < indpb:
            individual[i] = random.choice(valid_activations)

    # Перевіряємо особину після змін
    fix_individual(individual)
    return individual,

def custom_crossover(ind1, ind2):
    """Кросовер для розділених частин особини."""
    num_layers1 = len(ind1) // 2
    num_layers2 = len(ind2) // 2

    # Розділяємо особини на нейрони і активації
    neurons1, activations1 = ind1[:num_layers1], ind1[num_layers1:]
    neurons2, activations2 = ind2[:num_layers2], ind2[num_layers2:]

    # Перевірка довжини шарів нейронів для коректної роботи cxTwoPoint
    if len(neurons1) > 1 and len(neurons2) > 1:
        tools.cxTwoPoint(neurons1, neurons2)
    if len(activations1) > 1 and len(activations2) > 1:
        tools.cxTwoPoint(activations1, activations2)

    # Перевіряємо і збираємо особини назад
    ind1[:] = neurons1 + activations1
    ind2[:] = neurons2 + activations2
    fix_individual(ind1)
    fix_individual(ind2)

    return ind1, ind2

toolbox.register("evaluate", evaluate_nn)
toolbox.register("mate", custom_crossover)
toolbox.register("mutate", custom_mutate, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Основний процес ГА
if __name__ == "__main__":
    random.seed(42)

    # Початкова популяція
    population = toolbox.population(n=20)

    # Еволюція
    NGEN = 30
    CXPB, MUTPB = 0.5, 0.2

    print("Початкова популяція")
    for ind in population:
        print(ind)

    # Запуск алгоритму
    for gen in range(NGEN):
        print(f"\n=== Покоління {gen} ===")

        # Застосування еволюційного алгоритму
        population, logbook = algorithms.eaSimple(population, toolbox, cxpb=CXPB,
                                                  mutpb=MUTPB, ngen=1, verbose=False)

        # Оцінка особин:
        fits = [ind.fitness.values[0] for ind in population]
        length = len(population)
        mean = sum(fits) / length
        max_val = max(fits)
        print("  Мін %s" % min(fits), "  Макс %s" % max_val, "  Середнє %s" % mean)

        # Виведення найкращого рішення на цьому поколінні
        best_ind = tools.selBest(population, 1)[0]
        print("Найкраще рішення до цього моменту:", best_ind, best_ind.fitness.values)

    # Виведення найкращого рішення загалом
    best_ind = tools.selBest(population, 1)[0]
    print("\nОстаточно найкраще рішення:", best_ind, best_ind.fitness.values)

    final_model = evaluate_nn(best_ind)
    final_model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = final_model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16,
                              callbacks=[early_stopping], verbose=1)

    loss, accuracy = final_model.evaluate(X_test, y_test, verbose=0)
    print(f"Точність на тестових даних: {accuracy * 100:.2f}%")