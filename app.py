from flask import Flask, request
import subprocess

app = Flask(__name__)

@app.route('/')
def hola():
    return "Hola Mundo"

@app.route('/procesar_formulario')
def procesar_formulario():
     # Obtener el parámetro 'deseo' de la solicitud GET
    deseo = request.args.get('deseo')
    
    try:
        # Llamar al script de Python para verificar contenido inapropiado
        resultado_python = subprocess.check_output(['python', 'python_scripts/verificar_inapropiado.py', deseo])
        
        if resultado_python.strip() == b'True':
            return "True"
        else:
            # Agrega aquí la lógica para almacenar el deseo en la base de datos
            return "False"
    except subprocess.CalledProcessError as e:
        # Manejar error de subprocess
        return f"Error en el script de Python: {e}"
    except Exception as e:
        # Manejar otros errores
        return f"Error inesperado: {e}"

if __name__ == '__main__':
    app.run(debug=True)
