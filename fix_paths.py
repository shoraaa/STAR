import os
import re

def fix_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if '/home/shora/Research/NRS_Survey/NRS/' in content:
                    content = content.replace('/home/shora/Research/NRS_Survey/NRS/', '/home/shora/Research/STAR/survey/')
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"Fixed {path}")

fix_directory('survey/NRS')
