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
        continue
    
    with open(target, 'r') as f:
        content = f.read()

    # The file has things like:
    # print(f"Instance {\1}, dim={dimension}, length={\3}, optimal={\2}, gap={\4}%, time={\5}s")
    # or
    # self.logger.info(f"Instance {\1}, dim={dimension}, length={\3}, optimal={\2}, gap={\4}%, time={\5}s")

    # Let's fix those literal strings with the original variable names!
    # \1 -> name
    # \2 -> optimal/optimal_score/opt_len depending on file
    # \3 -> score/tour_len
    # \4 -> gap/no_aug_gap
    # \5 -> inst_time

    if "TSPTester_LIB_Survey.py" in target or "CVRPTester_LIB_Survey.py" in target or "test_vrplib_survey.py" in target or "test_tsplib_survey.py" in target or "test_cvrplib_survey.py" in target:
        if "ELG" in target or "ReLD" in target:
            content = re.sub(
                r'print\(f"Instance \{\\1\}, dim=\{dimension\}, length=\{\\3\}, optimal=\{\\2\}, gap=\{\\4\}%, time=\{\\5\}s"\)',
                r'print(f"Instance {name}, dim={dimension}, length={score}, optimal={optimal}, gap={no_aug_gap}%, time={inst_time}s")',
                content
            )
        else:
            content = re.sub(
                r'self\.logger\.info\(f"Instance \{\\1\}, dim=\{dimension\}, length=\{\\3\}, optimal=\{\\2\}, gap=\{\\4\}%, time=\{\\5\}s"\)',
                r'self.logger.info(f"Instance {name}, dim={dimension}, length={score}, optimal={optimal}, gap={no_aug_gap}%, time={inst_time}s")',
                content
            )
    elif "test_function_survey.py" in target:
        content = re.sub(
            r'print\(f"Instance \{\\1\}, dim=\{dimension\}, length=\{\\3\}, optimal=\{\\2\}, gap=\{\\4\}%, time=\{\\5\}s"\)',
            r'print(f"Instance {name}, dim={dimension}, length={tour_len}, optimal={opt_len}, gap={gap}%, time={inst_time}s")',
            content
        )
    elif "neuopt_survey_lib.py" in target:
        content = re.sub(
            r'print\(f"Instance \{\\1\}, dim=\{dimension\}, length=\{\\3\}, optimal=\{\\2\}, gap=\{\\4\}%, time=\{\\5\}s"\)',
            r'print(f"Instance {name}, dim={dimension}, length={score}, optimal={optimal_score}, gap={gap}%, time={inst_time}s")',
            content
        )

    with open(target, 'w') as f:
        f.write(content)
