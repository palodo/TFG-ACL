# Asistente de Diagnóstico Web (LCA)

Esta carpeta contiene la aplicación web para interactuar visualmente con el clasificador de roturas de LCA.

## 📁 Estructura

* **`frontend/`**: Proyecto en React (Vite) para la interfaz de usuario.
* **`server.js`**: Servidor Node.js (Express) que expone la API y sirve la aplicación compilada.
* **`start_server.py`**: Script automatizado para instalar dependencias, compilar el frontend y arrancar el servidor en un solo paso.

## 🚀 Cómo ejecutar la aplicación

Desde la raíz del proyecto o desde esta carpeta, ejecuta:

```bash
python start_server.py
```

El script se encargará de compilar la interfaz de React e iniciar el servidor Node.js localmente en el puerto `5000`. Podrás abrirla en tu navegador en:
`http://localhost:5000/`
