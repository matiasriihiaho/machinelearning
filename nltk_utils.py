import subprocess
from textblob import TextBlob
cmd = ['python3','-m','textblob.download_corpora']
#cmd = 'python -m textblob.download_corpora'
subprocess.run(cmd)
print("Working")