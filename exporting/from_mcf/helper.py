import os
import platform
from subprocess import Popen, PIPE


def get_r_home():
    os_name = platform.system()
    r_home = None

    if os_name == 'Windows':
        r_base_path = r"C:\Program Files\R"
        if os.path.exists(r_base_path):
            versions = sorted([d for d in os.listdir(r_base_path) if os.path.isdir(os.path.join(r_base_path, d))],
                              reverse=True)
            if versions:
                r_home = os.path.join(r_base_path, versions[0])
    elif os_name == 'Linux':
        if os.path.exists('/usr/local/lib/R'):
            r_home = '/usr/local/lib/R'
        elif os.path.exists('/usr/lib/R'):
            r_home = '/usr/lib/R'
    elif os_name == 'Darwin':  # macOS
        r_home = '/Library/Frameworks/R.framework/Resources'

    # Adjusting subprocess call to handle the R workspace image directive
    if not r_home or not os.path.exists(r_home):
        try:
            process = Popen(['R', 'RHOME'], stdout=PIPE, stderr=PIPE)
            stdout, stderr = process.communicate()
            r_home_detected = stdout.decode('utf-8').split("\n")[0].strip()
            if r_home_detected and os.path.exists(r_home_detected):
                r_home = r_home_detected
        except Exception as e:
            print(f"An error occurred while detecting R_HOME: {str(e)}")

    if r_home and os.path.exists(r_home):
        return r_home
    else:
        raise EnvironmentError("R_HOME could not be determined. Please set it manually or ensure R is installed.")
