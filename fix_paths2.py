import os

def fix_directory(directory):
    base_survey_dir = os.path.abspath('survey')
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                modified = False
                
                if '/home/shora/Research/STAR/survey/' in content:
                    rel_to_survey = os.path.relpath(base_survey_dir, os.path.abspath(root))
                    content = content.replace('/home/shora/Research/STAR/survey/', rel_to_survey + '/')
                    modified = True
                
                if modified:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"Fixed {path}")

fix_directory('survey/NRS')
