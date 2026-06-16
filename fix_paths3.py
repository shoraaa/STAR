import os
import re

def fix_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                modified = False
                
                # Replace any sequence of ../ followed by Construction/ or Improvement/
                new_content = re.sub(r'((?:\.\./)+)Construction/', r'\g<1>NRS/Construction/', content)
                new_content = re.sub(r'((?:\.\./)+)Improvement/', r'\g<1>NRS/Improvement/', new_content)
                
                if new_content != content:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"Fixed {path}")

fix_directory('survey/NRS')
