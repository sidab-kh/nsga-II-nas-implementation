import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# Importation de tes APIs (assure-toi que les noms de fichiers correspondent à ton projet)
from nas_201_api.api_201 import NASBench201API
# Assure-toi que hw_nas_bench_api.py est bien dans le même dossier ou ajuste l'import
from hw_nas_bench_api.hw_nas_bench_api import HWNASBenchAPI 

# ---------------------------------------------------------------------------
# 1. Définition du problème multi-objectif
# ---------------------------------------------------------------------------
class NASBenchMultiObjective(ElementwiseProblem):
    def __init__(self, api_201, api_hw, dataset="cifar10", hw_metric="edgegpu_latency"):
        # n_var = 1 : On cherche un seul index d'architecture
        # n_obj = 2 : Latence et Erreur
        # xl, xu : Bornes de l'espace de recherche (NAS-Bench-201 possède 15625 archis, donc index 0 à 15624)
        super().__init__(n_var=1, 
                         n_obj=2, 
                         n_ieq_constr=0, 
                         xl=np.array([0]), 
                         xu=np.array([15624]), 
                         vtype=int)
        
        self.api_201 = api_201
        self.api_hw = api_hw
        self.dataset = dataset
        self.hw_metric = hw_metric

    def _evaluate(self, x, out, *args, **kwargs):
        arch_idx = int(x[0]) # PyMoo passe un array numpy, on cast en int
        
        # --- Objectif 1 : Latence (à minimiser) ---
        hw_metrics = self.api_hw.query_by_index(arch_idx, self.dataset)
        latency = hw_metrics[self.hw_metric]
        
        # --- Objectif 2 : Erreur (à minimiser -> équivaut à maximiser l'accuracy) ---
        # hp='200' pour récupérer la meilleure précision après 200 epochs
        # is_random=False pour obtenir la moyenne des runs et éviter la variance
        info_201 = self.api_201.get_more_info(arch_idx, self.dataset, hp='200', is_random=False)
        accuracy = info_201['test-accuracy']
        error = 100.0 - accuracy 
        
        # Enregistrement des objectifs
        out["F"] = [latency, error]

# ---------------------------------------------------------------------------
# 2. Exécution de l'algorithme
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Chargement des bases de données...")
    # Remplace les chemins par les bons chemins absolus ou relatifs vers tes fichiers .pth et .pickle
    api_201 = NASBench201API('NAS-Bench-201-v1_1-096897.pth') 
    api_hw = HWNASBenchAPI('./HW-NAS-Bench-v1_0.pickle', search_space='nasbench201')
    
    # Paramétrage : Choix du dataset et du hardware cible (ex: Edge GPU, Edge TPU, FPGA, etc.)
    DATASET = "cifar10"
    TARGET_HARDWARE = "edgegpu_latency"
    
    print(f"Initialisation du problème pour {DATASET} optimisant l'accuracy et {TARGET_HARDWARE}...")
    problem = NASBenchMultiObjective(api_201, api_hw, dataset=DATASET, hw_metric=TARGET_HARDWARE)
    
    # Configuration de NSGA-II
    # On utilise RoundingRepair car on cherche un index entier discret
    algorithm = NSGA2(
        pop_size=50, # Taille de la population
        n_offsprings=50,
        crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
        mutation=PM(prob=0.1, eta=20, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True
    )
    
    print("Démarrage de la recherche NSGA-II (peut prendre quelques instants)...")
    res = minimize(problem,
                   algorithm,
                   ("n_gen", 40), # Nombre de générations (critère d'arrêt)
                   seed=42,
                   verbose=True)
    
    # ---------------------------------------------------------------------------
    # 3. Affichage des résultats (Front de Pareto)
    # ---------------------------------------------------------------------------
    print("\nRecherche terminée !")
    print(f"Nombre d'architectures trouvées sur le front de Pareto : {len(res.X)}")
    
    print("\nMeilleures architectures (Front de Pareto) :")
    print(f"{'Index Arch':<12} | {'Latence (ms)':<15} | {'Accuracy (%)':<15}")
    print("-" * 48)
    
    # Trier les résultats par latence pour un affichage propre
    sorted_indices = np.argsort(res.F[:, 0])
    
    for i in sorted_indices:
        arch_id = int(res.X[i][0])
        latency = res.F[i, 0]
        acc = 100.0 - res.F[i, 1] # On reconvertit l'erreur en accuracy
        print(f"{arch_id:<12} | {latency:<15.4f} | {acc:<15.2f}")

    # Plot du front de Pareto
    plot = Scatter(title="Front de Pareto : Latence vs Erreur", 
                   labels=["Latence (ms)", "Erreur (100 - Acc%)"])
    plot.add(res.F, color="red")
    plot.show()