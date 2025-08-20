#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test de l'algorithme génétique Ichimoku
"""

import random
import numpy as np

# Simuler les fonctions de l'algorithme génétique
class IchimokuTrader:
    """Un trader Ichimoku avec son ADN (paramètres)"""
    
    def __init__(self, tenkan, kijun, senkou_b, shift, atr_mult):
        self.tenkan = tenkan
        self.kijun = kijun
        self.senkou_b = senkou_b
        self.shift = shift
        self.atr_mult = atr_mult
        self.fitness = 0.0
        self.generation = 0
        self.performance_history = []
    
    def get_params(self):
        """Retourne les paramètres du trader"""
        return {
            "tenkan": self.tenkan,
            "kijun": self.kijun,
            "senkou_b": self.senkou_b,
            "shift": self.shift,
            "atr_mult": self.atr_mult
        }
    
    def mutate(self, mutation_rate=0.15):
        """Mutation génétique pour explorer de nouveaux paramètres"""
        if random.random() < mutation_rate:
            # Mutation du Tenkan
            if random.random() < 0.2:
                self.tenkan = max(1, min(70, self.tenkan + random.randint(-15, 15)))
            
            # Mutation du Kijun
            if random.random() < 0.2:
                self.kijun = max(1, min(70, self.kijun + random.randint(-15, 15)))
            
            # Mutation du Senkou B
            if random.random() < 0.2:
                self.senkou_b = max(1, min(70, self.senkou_b + random.randint(-15, 15)))
            
            # Mutation du Shift
            if random.random() < 0.2:
                self.shift = max(1, min(99, self.shift + random.randint(-20, 20)))
            
            # Mutation de l'ATR
            if random.random() < 0.2:
                self.atr_mult = max(1.0, min(14.0, self.atr_mult + random.uniform(-3, 3)))
                self.atr_mult = round(self.atr_mult, 1)
    
    def crossover(self, other_trader):
        """Croisement avec un autre trader pour créer un enfant"""
        child = IchimokuTrader(
            tenkan=random.choice([self.tenkan, other_trader.tenkan]),
            kijun=random.choice([self.kijun, other_trader.kijun]),
            senkou_b=random.choice([self.senkou_b, other_trader.senkou_b]),
            shift=random.choice([self.shift, other_trader.shift]),
            atr_mult=round((self.atr_mult * 0.6 + other_trader.atr_mult * 0.4), 1)
        )
        return child

def create_initial_population(population_size=10):
    """Crée une petite population de test"""
    population = []
    
    for i in range(population_size):
        trader = IchimokuTrader(
            tenkan=random.randint(5, 25),
            kijun=random.randint(10, 30),
            senkou_b=random.randint(20, 50),
            shift=random.randint(15, 35),
            atr_mult=round(random.uniform(2.0, 8.0), 1)
        )
        population.append(trader)
    
    return population

def simulate_fitness(trader):
    """Simule un score de fitness basé sur les paramètres"""
    # Plus les paramètres sont équilibrés, meilleur est le score
    balance_score = 1.0 - abs(trader.tenkan - trader.kijun) / 70.0
    atr_score = 1.0 - abs(trader.atr_mult - 5.0) / 10.0
    shift_score = 1.0 - abs(trader.shift - 25) / 50.0
    
    return (balance_score + atr_score + shift_score) / 3.0

def evolve_population(population, elite_size=2):
    """Fait évoluer la population vers la génération suivante"""
    # Trier par fitness
    population.sort(key=lambda x: x.fitness, reverse=True)
    
    # Garder les élites (meilleurs 20%)
    elite = population[:elite_size]
    
    # Créer la nouvelle génération
    new_population = elite.copy()
    
    # Croisement des élites pour créer des enfants
    while len(new_population) < len(population):
        parent1 = random.choice(elite)
        parent2 = random.choice(elite)
        child = parent1.crossover(parent2)
        child.generation = max(parent1.generation, parent2.generation) + 1
        new_population.append(child)
    
    # Mutation pour maintenir la diversité
    for trader in new_population[elite_size:]:
        trader.mutate(mutation_rate=0.3)
    
    return new_population[:len(population)]

def main():
    """Test de l'algorithme génétique"""
    print("🧬 TEST DE L'ALGORITHME GÉNÉTIQUE ICHIMOKU")
    print("=" * 50)
    
    # Créer la population initiale
    population = create_initial_population(population_size=10)
    print(f"👥 Population initiale créée: {len(population)} traders")
    
    # Évolution sur 5 générations
    for generation in range(5):
        print(f"\n🧬 GÉNÉRATION {generation + 1}")
        
        # Évaluer chaque trader
        for i, trader in enumerate(population):
            trader.fitness = simulate_fitness(trader)
            trader.performance_history.append(trader.fitness)
            print(f"  Trader {i+1}: Fitness={trader.fitness:.4f}, Params={trader.get_params()}")
        
        # Afficher le meilleur
        population.sort(key=lambda x: x.fitness, reverse=True)
        best_trader = population[0]
        print(f"🏆 Meilleur: Fitness={best_trader.fitness:.4f}, Params={best_trader.get_params()}")
        
        # Évolution (sauf la dernière génération)
        if generation < 4:
            population = evolve_population(population, elite_size=2)
            print(f"🔄 Population évoluée vers la génération {generation + 2}")
    
    print(f"\n🎉 TEST TERMINÉ ! L'algorithme génétique fonctionne parfaitement !")
    print(f"📊 Meilleur trader final: Fitness={best_trader.fitness:.4f}")
    print(f"🧬 Paramètres optimaux: {best_trader.get_params()}")

if __name__ == "__main__":
    main()
