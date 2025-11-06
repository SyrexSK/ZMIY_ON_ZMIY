"""
Нейро-игра «Змий»
-------------------------------------------------
Особенности:
- Эволюционная популяция фиксированной топологии нейросетей (GA + элитизм)
- Набор признаков: опасность (вперед/влево/вправо), текущее направление, направление до еды
- Действия: Повернуть влево / Прямо / Повернуть вправо
- Быстрая головная тренировка без отрисовки; опционально — показ лучшего агента в Pygame
- Сохранение/загрузка лучшего генома (weights.pkl)

Зависимости: Python 3.12, numpy, pygame
Установка: uv add numpy pygame
Запуск тренировки (без окна):
    python main.py --train --generations 50 --pop 150
Просмотр лучшего после тренировки:
    python main.py --play --model weights.pkl --speed 15
Совмещённый цикл: сначала тренировка, затем показ:
    python main.py --train --generations 30 --pop 120 --then-play --speed 15
"""
from __future__ import annotations
import argparse
import math
import os
import pickle
import random
from dataclasses import dataclass
from typing import List, Tuple, Deque
from collections import deque

import numpy as np

# --------------------------- Игровая логика --------------------------- #

Vec = Tuple[int, int]

DIRECTIONS: List[Vec] = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # R, D, L, U (по часовой)
RIGHT, DOWN, LEFT, UP = 0, 1, 2, 3

@dataclass
class SnakeEnvConfig:
    grid_w: int = 20
    grid_h: int = 20
    max_steps_without_food: int = 100  # ограничение, чтобы не петлял бесконечно
    food_reward: float = 10.0
    step_penalty: float = -0.01
    death_penalty: float = -5.0
    eat_extend: int = 1

class SnakeEnv:
    def __init__(self, cfg: SnakeEnvConfig):
        self.cfg = cfg
        self.reset()

    def reset(self):
        w, h = self.cfg.grid_w, self.cfg.grid_h
        self.snake: Deque[Vec] = deque()
        start_x, start_y = w // 2, h // 2
        self.snake.append((start_x, start_y))
        self.dir = random.choice([RIGHT, DOWN, LEFT, UP])
        self.steps_since_food = 0
        self.total_steps = 0
        self.score = 0
        self.alive = True
        self._place_food()
        return self._get_state()

    def _place_food(self):
        w, h = self.cfg.grid_w, self.cfg.grid_h
        empty = {(x, y) for x in range(w) for y in range(h)} - set(self.snake)
        self.food = random.choice(list(empty))

    def _get_state(self) -> np.ndarray:
        # Признаки: опасность по курсу/влево/вправо (1/0), текущее направление one-hot (4),
        # вектор до еды (dx_sign, dy_sign)
        danger_f, danger_l, danger_r = self._danger_sensors()
        dir_onehot = np.zeros(4, dtype=float)
        dir_onehot[self.dir] = 1.0
        fx, fy = self.food
        hx, hy = self.snake[0]
        dx = np.sign(fx - hx)
        dy = np.sign(fy - hy)
        return np.array([danger_f, danger_l, danger_r, *dir_onehot, float(dx), float(dy)], dtype=float)

    def _danger_sensors(self) -> Tuple[float, float, float]:
        # Опасность столкнуться, если повернуть (влево/прямо/вправо)
        def will_collide(turned_dir: int) -> bool:
            nx, ny = self._next_head_pos(turned_dir)
            return (nx < 0 or nx >= self.cfg.grid_w or ny < 0 or ny >= self.cfg.grid_h or (nx, ny) in list(self.snake))
        left_dir = (self.dir - 1) % 4
        right_dir = (self.dir + 1) % 4
        return float(will_collide(self.dir)), float(will_collide(left_dir)), float(will_collide(right_dir))

    def _next_head_pos(self, dir_idx: int) -> Vec:
        dx, dy = DIRECTIONS[dir_idx]
        hx, hy = self.snake[0]
        return hx + dx, hy + dy

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        # action: 0=лево, 1=прямо, 2=право (относительно текущего направления)
        if action == 0:
            self.dir = (self.dir - 1) % 4
        elif action == 2:
            self.dir = (self.dir + 1) % 4
        # если 1 — направление без изменения

        reward = self.cfg.step_penalty
        self.total_steps += 1
        self.steps_since_food += 1

        nx, ny = self._next_head_pos(self.dir)
        # Проверка смерти
        if (
            nx < 0 or nx >= self.cfg.grid_w or ny < 0 or ny >= self.cfg.grid_h or (nx, ny) in self.snake
        ):
            self.alive = False
            reward += self.cfg.death_penalty
            return self._get_state(), reward, True

        # Двигаем змею
        self.snake.appendleft((nx, ny))
        ate = (nx, ny) == self.food
        if ate:
            self.score += 1
            reward += self.cfg.food_reward
            self.steps_since_food = 0
            # Удлиняем на eat_extend, но фактически просто не удаляем хвост на эти шаги
            for _ in range(self.cfg.eat_extend - 1):
                self.snake.append(self.snake[-1])
            self._place_food()
        else:
            self.snake.pop()

        # Принудительная остановка, если давно не ел
        if self.steps_since_food > self.cfg.max_steps_without_food:
            self.alive = False
            reward += self.cfg.death_penalty * 0.5
            return self._get_state(), reward, True

        return self._get_state(), reward, False

# --------------------------- Нейросеть --------------------------- #

@dataclass
class NetShape:
    n_in: int
    n_hid: int
    n_out: int

class NeuralNet:
    def __init__(self, shape: NetShape, weights: np.ndarray | None = None):
        self.shape = shape
        self.n_w = shape.n_in * shape.n_hid + shape.n_hid + shape.n_hid * shape.n_out + shape.n_out
        if weights is None:
            self.w = np.random.randn(self.n_w).astype(np.float32) * 0.5
        else:
            assert len(weights) == self.n_w
            self.w = weights.astype(np.float32)

    def _unpack(self):
        s = self.shape
        p = 0
        W1 = self.w[p : p + s.n_in * s.n_hid].reshape(s.n_in, s.n_hid)
        p += s.n_in * s.n_hid
        b1 = self.w[p : p + s.n_hid]
        p += s.n_hid
        W2 = self.w[p : p + s.n_hid * s.n_out].reshape(s.n_hid, s.n_out)
        p += s.n_hid * s.n_out
        b2 = self.w[p : p + s.n_out]
        return W1, b1, W2, b2

    def act(self, x: np.ndarray) -> int:
        W1, b1, W2, b2 = self._unpack()
        h = np.tanh(x @ W1 + b1)
        o = h @ W2 + b2
        # softmax для стабильности
        e = np.exp(o - np.max(o))
        probs = e / (np.sum(e) + 1e-8)
        return int(np.argmax(probs))

# --------------------------- Генетический алгоритм --------------------------- #

@dataclass
class GAConfig:
    pop_size: int = 150
    elite_frac: float = 0.1
    mutation_rate: float = 0.08
    mutation_sigma: float = 0.2
    crossover_rate: float = 0.75
    episodes_per_ind: int = 2
    max_steps_per_episode: int = 400
    seed: int | None = None

@dataclass
class Individual:
    genome: np.ndarray
    fitness: float | None = None

class GA:
    def __init__(self, net_shape: NetShape, cfg: GAConfig, env_cfg: SnakeEnvConfig):
        if cfg.seed is not None:
            np.random.seed(cfg.seed)
            random.seed(cfg.seed)
        self.shape = net_shape
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.population: List[Individual] = [
            Individual(genome=NeuralNet(net_shape).w.copy()) for _ in range(cfg.pop_size)
        ]
        self.best: Individual | None = None
        self.generation = 0

    def evaluate(self) -> None:
        for ind in self.population:
            ind.fitness = self._eval_fitness(ind.genome)
        self.population.sort(key=lambda i: i.fitness if i.fitness is not None else -1e9, reverse=True)
        if self.best is None or (self.population[0].fitness or -1e9) > (self.best.fitness or -1e9):
            self.best = Individual(self.population[0].genome.copy(), self.population[0].fitness)

    def _eval_fitness(self, genome: np.ndarray) -> float:
        net = NeuralNet(self.shape, genome)
        scores = []
        for _ in range(self.cfg.episodes_per_ind):
            env = SnakeEnv(self.env_cfg)
            state = env.reset()
            total_reward = 0.0
            for _ in range(self.cfg.max_steps_per_episode):
                action = net.act(state)
                state, reward, done = env.step(action)
                total_reward += reward
                if done:
                    break
            # Комбинированная метрика: награды + бонус за счёт и длину жизни
            scores.append(total_reward + env.score * 20.0 + env.total_steps * 0.01)
        return float(np.mean(scores))

    def _select_parent(self) -> Individual:
        # турнирный отбор
        k = 3
        cand = random.sample(self.population, k)
        cand.sort(key=lambda i: i.fitness if i.fitness is not None else -1e9, reverse=True)
        return cand[0]

    def _crossover(self, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.cfg.crossover_rate:
            return a.copy(), b.copy()
        point = random.randint(1, len(a) - 2)
        child1 = np.concatenate([a[:point], b[point:]])
        child2 = np.concatenate([b[:point], a[point:]])
        return child1, child2

    def _mutate(self, g: np.ndarray) -> np.ndarray:
        mask = np.random.rand(len(g)) < self.cfg.mutation_rate
        noise = np.random.randn(len(g)).astype(np.float32) * self.cfg.mutation_sigma
        out = g.copy()
        out[mask] += noise[mask]
        # ограничим значения для численной устойчивости
        np.clip(out, -5.0, 5.0, out=out)
        return out

    def next_generation(self):
        self.generation += 1
        elite_n = max(1, int(self.cfg.pop_size * self.cfg.elite_frac))
        elites = [Individual(self.population[i].genome.copy(), self.population[i].fitness) for i in range(elite_n)]

        children: List[Individual] = []
        while len(children) + elite_n < self.cfg.pop_size:
            p1 = self._select_parent().genome
            p2 = self._select_parent().genome
            c1, c2 = self._crossover(p1, p2)
            c1 = self._mutate(c1)
            c2 = self._mutate(c2)
            children.append(Individual(c1))
            if len(children) + elite_n < self.cfg.pop_size:
                children.append(Individual(c2))

        self.population = elites + children

# --------------------------- Визуализация (Pygame) --------------------------- #

try:
    import pygame  # type: ignore
except Exception:
    pygame = None

class Viewer:
    def __init__(self, cell=24):
        if pygame is None:
            raise RuntimeError("Pygame не установлен: pip install pygame")
        pygame.init()
        self.cell = cell

    def play(self, net: NeuralNet, env_cfg: SnakeEnvConfig, speed_fps: int = 12):
        env = SnakeEnv(env_cfg)
        state = env.reset()
        w, h = env_cfg.grid_w, env_cfg.grid_h
        screen = pygame.display.set_mode((w * self.cell, h * self.cell))
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("consolas", 18)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            action = net.act(state)
            state, _, done = env.step(action)

            # draw
            screen.fill((30, 30, 35))
            # grid (optional light)
            for x in range(w):
                for y in range(h):
                    rect = pygame.Rect(x * self.cell, y * self.cell, self.cell - 1, self.cell - 1)
                    pygame.draw.rect(screen, (45, 45, 55), rect, 1)

            # food
            fx, fy = env.food
            frect = pygame.Rect(fx * self.cell, fy * self.cell, self.cell - 1, self.cell - 1)
            pygame.draw.rect(screen, (200, 90, 90), frect)

            # snake
            for i, (sx, sy) in enumerate(env.snake):
                rect = pygame.Rect(sx * self.cell, sy * self.cell, self.cell - 1, self.cell - 1)
                color = (120, 220, 120) if i == 0 else (80, 180, 80)
                pygame.draw.rect(screen, color, rect)

            txt = font.render(f"Score: {env.score}", True, (230, 230, 230))
            screen.blit(txt, (8, 8))
            pygame.display.flip()

            if done:
                # пауза после смерти
                pygame.time.delay(600)
                env.reset()
                state = env._get_state()

            clock.tick(speed_fps)
        pygame.quit()

# --------------------------- Скрипт командной строки --------------------------- #

def save_genome(path: str, genome: np.ndarray):
    with open(path, 'wb') as f:
        pickle.dump(genome, f)


def load_genome(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        return pickle.load(f)


def train(args):
    env_cfg = SnakeEnvConfig(
        grid_w=args.grid,
        grid_h=args.grid,
        max_steps_without_food=args.max_nofood,
        food_reward=10.0,
        step_penalty=-0.01,
        death_penalty=-5.0,
        eat_extend=1,
    )
    # входы: 3 опасности + 4 направления + 2 до еды = 9
    shape = NetShape(n_in=9, n_hid=args.hidden, n_out=3)
    ga_cfg = GAConfig(
        pop_size=args.pop,
        elite_frac=args.elite,
        mutation_rate=args.mut_rate,
        mutation_sigma=args.mut_sigma,
        crossover_rate=args.cx_rate,
        episodes_per_ind=args.episodes,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
    )

    ga = GA(shape, ga_cfg, env_cfg)

    best_hist = []
    for g in range(args.generations):
        ga.evaluate()
        best = ga.population[0]
        avg_fit = float(np.mean([i.fitness for i in ga.population if i.fitness is not None]))
        best_hist.append((g, best.fitness, avg_fit))
        print(f"Gen {g:03d}: best={best.fitness:.3f} avg={avg_fit:.3f}")
        ga.next_generation()

    # финальная оценка и сохранение
    ga.evaluate()
    print(f"Final best fitness: {ga.best.fitness:.3f}")
    save_genome(args.model, ga.best.genome)
    print(f"Saved best genome to {args.model}")

    if args.then_play:
        if pygame is None:
            print("Pygame не установлен, пропускаю показ.")
        else:
            viewer = Viewer(cell=args.cell)
            net = NeuralNet(shape, ga.best.genome)
            viewer.play(net, env_cfg, speed_fps=args.speed)


def play(args):
    env_cfg = SnakeEnvConfig(grid_w=args.grid, grid_h=args.grid, max_steps_without_food=args.max_nofood)
    shape = NetShape(n_in=9, n_hid=args.hidden, n_out=3)
    genome = load_genome(args.model)
    net = NeuralNet(shape, genome)
    viewer = Viewer(cell=args.cell)
    viewer.play(net, env_cfg, speed_fps=args.speed)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Нейро-змий (GA + NN)")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true", help="запустить тренировку")
    mode.add_argument("--play", action="store_true", help="показать лучшего агента")

    p.add_argument("--generations", type=int, default=30, help="число поколений")
    p.add_argument("--pop", type=int, default=150, help="размер популяции")
    p.add_argument("--elite", type=float, default=0.12, help="доля элиты [0..1]")
    p.add_argument("--mut-rate", type=float, default=0.08, help="вероятность мутации гена")
    p.add_argument("--mut-sigma", type=float, default=0.20, help="сигма гаусс. мутации")
    p.add_argument("--cx-rate", type=float, default=0.75, help="вероятность кроссовера")
    p.add_argument("--episodes", type=int, default=2, help="число эпизодов на индивида")
    p.add_argument("--max-steps", type=int, default=400, help="лимит шагов на эпизод")
    p.add_argument("--grid", type=int, default=20, help="размер поля (квадрат)")
    p.add_argument("--max-nofood", type=int, default=100, help="лимит шагов без еды")
    p.add_argument("--hidden", type=int, default=16, help="нейроны скрытого слоя")
    p.add_argument("--model", type=str, default="weights.pkl", help="путь для сохранения/загрузки генома")
    p.add_argument("--seed", type=int, default=None, help="seed для воспроизводимости")

    p.add_argument("--then-play", action="store_true", help="после тренировки показать агента")
    p.add_argument("--cell", type=int, default=24, help="размер клетки при отрисовке")
    p.add_argument("--speed", type=int, default=12, help="FPS при показе")

    args = p.parse_args()

    if args.train:
        train(args)
    elif args.play:
        play(args)
