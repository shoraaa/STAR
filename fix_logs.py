import os
import re

targets = [
    "survey/NRS/Construction/single-stage/appending/4_ICAM/ICAM_TSP/TSPTester_LIB_Survey.py",
    "survey/NRS/Construction/single-stage/appending/4_ICAM/ICAM_CVRP/CVRPTester_LIB_Survey.py",
    "survey/NRS/Construction/single-stage/appending/5_ELG/TSP/test_tsplib_survey.py",
    "survey/NRS/Construction/single-stage/appending/5_ELG/CVRP/test_vrplib_survey.py",
    "survey/NRS/Construction/single-stage/appending/6_INViT/test_function_survey.py",
    "survey/NRS/Construction/single-stage/appending/7_L2R/TSP/TSPTester_LIB_Survey.py",
    "survey/NRS/Construction/single-stage/appending/7_L2R/CVRP/CVRPTester_LIB_Survey.py",
    "survey/NRS/Construction/single-stage/appending/9_ReLD/CVRP/test_cvrplib_survey.py",
    "survey/NRS/Improvement/single_solution_based/small neighborhood/sequential/NeuOpt/neuopt_survey_lib.py",
]

for target in targets:
    if not os.path.exists(target):
        print(f"Missing {target}")
        continue
    
    with open(target, 'r') as f:
        content = f.read()

    # Pattern for ICAM/L2R TSP
    content = re.sub(
        r'self\.logger\.info\("Instance name: \{\}, optimal score: \{:\.4f\}"\.format\((name), (optimal)\)\)\s*self\.logger\.info\("No aug score:\{:\.3f\}, No aug gap:\{:\.3f\}%"\.format\((score), (no_aug_gap)\)\)\s*.*self\.logger\.info\(f"Instance time [^:]+: \{(inst_time):\.3f\}s"\)',
        r'self.logger.info(f"Instance {\\1}, dim={dimension}, length={\\3}, optimal={\\2}, gap={\\4}%, time={\\5}s")',
        content, flags=re.DOTALL
    )

    # Pattern for ICAM/L2R CVRP
    content = re.sub(
        r'self\.logger\.info\("Instance name: \{\}, optimal score: \{:\.4f\}"\.format\((name), (optimal)\)\)\s*self\.logger\.info\("No aug score:\{:\.3f\}, No aug gap:\{:\.3f\}%"\.format\((score), (no_aug_gap)\)\)\s*.*self\.logger\.info\(f"Instance time: \{(inst_time):\.3f\}s"\)',
        r'self.logger.info(f"Instance {\\1}, dim={dimension}, length={\\3}, optimal={\\2}, gap={\\4}%, time={\\5}s")',
        content, flags=re.DOTALL
    )

    # Pattern for ELG
    content = re.sub(
        r'print\("Instance name: \{\}, optimal score: \{:\.4f\}"\.format\((name), (optimal)\)\)\s*print\("No aug score:\{:\.3f\}, No aug gap:\{:\.3f\}%"\.format\((score), (no_aug_gap)\)\)\s*.*print\(f"Instance time [^:]+: \{(inst_time):\.3f\}s"\)',
        r'print(f"Instance {\\1}, dim={dimension}, length={\\3}, optimal={\\2}, gap={\\4}%, time={\\5}s")',
        content, flags=re.DOTALL
    )

    # Pattern for INViT
    content = re.sub(
        r'print\("Instance name: \{\}, optimal score: \{:\.4f\}"\.format\((name), (opt_len)\)\)\s*print\("score:\{:\.3f\}, gap:\{:\.3f\}%"\.format\((tour_len), (gap)\)\)\s*.*print\(f"Instance time: \{(inst_time):\.3f\}s"\)',
        r'print(f"Instance {\\1}, dim={dimension}, length={\\3}, optimal={\\2}, gap={\\4}%, time={\\5}s")',
        content, flags=re.DOTALL
    )

    # Pattern for NeuOpt
    content = re.sub(
        r'print\("Instance name: \{\}, optimal score: \{:\.4f\}"\.format\((name), (optimal_score)\)\)\s*print\("score:\{:\.3f\}, gap:\{:\.3f\}%"\.format\((score), (gap)\)\)\s*.*print\(f"Instance time: \{(inst_time):\.3f\}s"\)',
        r'print(f"Instance {\\1}, dim={dimension}, length={\\3}, optimal={\\2}, gap={\\4}%, time={\\5}s")',
        content, flags=re.DOTALL
    )
    
    # Pattern for ReLD
    content = re.sub(
        r'print\("Instance name: \{\}, optimal score: \{:\.4f\}"\.format\((name), (optimal)\)\)\s*print\("No aug score:\{:\.3f\}, No aug gap:\{:\.3f\}%"\.format\((score), (no_aug_gap)\)\)\s*.*print\(f"Instance time: \{(inst_time):\.3f\}s"\)',
        r'print(f"Instance {\\1}, dim={dimension}, length={\\3}, optimal={\\2}, gap={\\4}%, time={\\5}s")',
        content, flags=re.DOTALL
    )

    with open(target, 'w') as f:
        f.write(content)
