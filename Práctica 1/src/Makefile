main:
	@echo Nuestro programa está escrito en el lenguaje Python, que es interpretado Por
	@echo lo tanto, no requiere de compilación alguna.

mp_help:
	@echo mp_help: explicación de los argumentos necesario para ejecutar el programa
	@echo
	@echo Para poder ejecutar correctamente el programa, hay que pasarle un archivo como argumento.
	@echo Como el programa esta escrito en python utilizaremos el intérprete \"python3\".
	@echo El programa principal es \"mainMP.py\", así que una correcta llamada al programa sería:
	@echo
	@echo -e '\t' \"python3 mainMP.py nombre_archivo.extension\"
	@echo
	@echo Con \"make mp_exec\" pueden ver un ejemplo de ejecución.

mp_exec:
	@echo un ejemplo de ejecución:
	python3 mainMP.py ../data/McCulloch_Pitts.txt

per_help:
	@echo per_help: explicación de los argumentos necesarios para ejecutar el programa perceptrón
	@echo
	@echo Existen tres modos para poder ejecutar correctamente el programa.
	@echo -Modo1: Los datos de train y test se obtienen del mismo archivo en función del porcentaje de entrenamiento.
	@echo -e '\t' \"python3 mainP.py nombre_archivo.extension porcentaje_entrenamiento\(1-100\) umbral tasa_aprendizaje max_epocas\"
	@echo
	@echo -Modo2: El porcentaje de train es 100% y por lo tanto los datos de train y test se obtienen del mismo fichero y se
	@echo cogen todos los datos para las dos particiones.
	@echo -e '\t' \"python3 mainP.py nombre_archivo.extension porcentaje_entrenamiento\(100\) umbral tasa_aprendizaje max_epocas\"
	@echo
	@echo -Modo3: Los datos de train se obtienen de un primer archivo y los de test de un segundo archivo, por lo tanto no requiere
	@echo de ningun porcentaje de entrenamiento.
	@echo -e '\t' \"python3 mainP.py nombre_archivo1.extension nombre_archivo2.extension umbral tasa_aprendizaje max_epocas\"
	@echo
	@echo La extension del archivo a leer que hemos elegido es \".in\" pero podría ser cualquier
	@echo otra como \".txt\"
	@echo
	@echo Con \"make per_exec\" pueden ver un ejemplo de ejecución.
	@echo Con \"make per_exec_puertas_logicas\" pueden ver otros ejemplos de ejecuciones con puertas lógicas.

per_exec:
	@echo Un ejemplo de ejecución:
	python3 mainP.py ../data/problema_real2.txt 75 0.2 0.3 2000

per_exec_puertas_logicas:
	@echo Tres ejemplos de ejecución en modo 2:
	python3 mainP.py ../data/nand.txt 100 0.2 0.1 100
	@echo
	python3 mainP.py ../data/nor.txt 100 0.2 0.1 100
	@echo
	python3 mainP.py ../data/xor.txt 100 0.2 0.1 100
	@echo
	@echo Ejemplo en modo 3
	python3 mainP.py ../data/problema_real2.txt ../data/problema_real2_no_etiquetados.txt 0.2 0.3 2000

ada_help:
	@echo ada_help: explicación de los argumentos necesarios para ejecutar el programa adaline
	@echo
	@echo Existen tres modos para poder ejecutar correctamente el programa.
	@echo -Modo1: Los datos de train y test se obtienen del mismo archivo en función del porcentaje de entrenamiento.
	@echo -e '\t' \"python3 mainP.py nombre_archivo.extension porcentaje_entrenamiento\(1-100\) umbral tolerancia tasa_aprendizaje max_epocas\"
	@echo
	@echo -Modo2: El porcentaje de train es 100% y por lo tanto los datos de train y test se obtienen del mismo fichero y se
	@echo cogen todos los datos para las dos particiones.
	@echo -e '\t' \"python3 mainP.py nombre_archivo.extension porcentaje_entrenamiento\(100\) umbral tolerancia tasa_aprendizaje max_epocas\"
	@echo
	@echo -Modo3: Los datos de train se obtienen de un primer archivo y los de test de un segundo archivo, por lo tanto no requiere
	@echo de ningun porcentaje de entrenamiento.
	@echo -e '\t' \"python3 mainP.py nombre_archivo1.extension nombre_archivo2.extension umbral tolerancia tasa_aprendizaje max_epocas\"
	@echo
	@echo La extension del archivo a leer que hemos elegido es \".in\" pero podría ser cualquier
	@echo otra como \".txt\"
	@echo
	@echo Con \"make ada_exec\" pueden ver un ejemplo de ejecución.
	@echo Con \"make ada_exec_puertas_logicas\" pueden ver otros ejemplos de ejecuciones con puertas lógicas.

ada_exec:
	@echo Un ejemplo de ejecución:
	python3 mainA.py ../data/problema_real2.txt 70 0.2 0.01 0.1 3000

ada_exec_puertas_logicas:
	@echo Tres ejemplos de ejecución en modo 2:
	python3 mainA.py ../data/nand.txt 100 0.2 0.01 0.1 100
	@echo
	python3 mainA.py ../data/nor.txt 100 0.2 0.01 0.1 100
	@echo
	python3 mainA.py ../data/xor.txt 100 0.2 0.01 0.1 100
	@echo
	@echo Ejemplo en modo 3
	python3 mainA.py ../data/problema_real2.txt ../data/problema_real2_no_etiquetados.txt 0.2 0.01 0.1 3000
