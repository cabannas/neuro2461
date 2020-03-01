mp_help:
	@echo mp_help: explicación de los argumentos necesario para ejecutar el programa
	@echo
	@echo Para poder ejecutar correctamente el programa, hay que pasarle un archivo como argumento.
	@echo Como el programa esta escrito en python utilizaremos el intérprete \"python3\".
	@echo El programa principal es \"mainMP.py\", así que una correcta llamada al programa sería:
	@echo
	@echo -e '\t' \"python3 mainMP.py nombre_archivo.extension\"
	@echo
	@echo Con \"make mp_exec\" pueden ver un ejemplo de ejecucción.

mp_exec:
	@echo un ejemplo de ejecución:
	python3 mainMP.py ../data/McCulloch_Pitts.txt

per_help:
	@echo per_help: explicación de los argumentos necesarios para ejecutar el programa perceptrón
	@echo
	@echo Existen tres modos para poder ejecutar correctamente el programa.
	@echo - Modo1: Los datos de train y test se obtienen del mismo archivo en función del porcentaje de entrenamiento.
	@echo -e '\t' \"python3 mainP.py nombre_archivo.extension porcentaje_entrenamiento umbral tasa_aprendizaje max_epocas\"
	@echo - Modo2: Los datos de train y test se obtienen del mismo fichero y se cogen todos los datos para las dos particiones.
	@echo -e '\t' \"python3 mainP.py nombre_archivo.extension umbral tasa_aprendizaje max_epocas\"
	@echo - Modo3: Los datos de train se obtienen del primer archivo y los de test del segundo.
	@echo -e '\t' \"python3 mainP.py nombre_archivo1.extension nombre_archivo2.extension umbral tasa_aprendizaje max_epocas\"
	@echo 
	@echo La extension del archivo a leer que hemos elegido es \".in\" pero podría ser cualquier
	@echo otra como \".txt\"
	@echo
	@echo Con \"make per_exec\" pueden ver un ejemplo de ejecucción.
	@echo Con \"make per_exec_puertas_logicas\" pueden ver otros ejemplos de ejecuciones con puertas lógicas.

per_exec:
	@echo un ejemplo de ejecución:
	python3 mainP.py ../data/problema_real2.txt 0.7 0.2 0.3 3000

per_exec_puertas_logicas:
	@echo tres ejemplos de ejecución en modo 2:
	python3 mainP.py ../data/nand.txt 0.2 0.1 100
	python3 mainP.py ../data/nor.txt 0.2 0.1 100
	python3 mainP.py ../data/xor.txt 0.2 0.1 100

ada_help:
	@echo per_help: explicación de los argumentos necesarios para ejecutar el programa adaline
	@echo
	@echo Existen tres modos para poder ejecutar correctamente el programa.
	@echo - Modo1: Los datos de train y test se obtienen del mismo archivo en función del porcentaje de entrenamiento.
	@echo -e '\t' \"python3 mainA.py nombre_archivo.extension porcentaje_entrenamiento umbral tolerancia tasa_aprendizaje max_epocas\"
	@echo - Modo2: Los datos de train y test se obtienen del mismo fichero y se cogen todos los datos para las dos particiones.
	@echo -e '\t' \"python3 mainA.py nombre_archivo.extension umbral tolerancia tasa_aprendizaje max_epocas\"
	@echo - Modo3: Los datos de train se obtienen del primer archivo y los de test del segundo.
	@echo -e '\t' \"python3 mainA.py nombre_archivo1.extension nombre_archivo2.extension umbral tolerancia tasa_aprendizaje max_epocas\"
	@echo
	@echo Con \"make per_exec\" pueden ver un ejemplo de ejecucción.
	@echo Con \"make per_exec_puertas_logicas\" pueden ver otros ejemplos de ejecuciones con puertas lógicas.

ada_exec:
	@echo un ejemplo de ejecución:
	python3 mainA.py ../data/problema_real2.txt 0.7 0.2 0.01 0.1 3000

ada_exec_puertas_logicas:
	@echo tres ejemplos de ejecución en modo 2:
	python3 mainA.py ../data/nand.txt 0.2 0.01 0.1 100
	python3 mainA.py ../data/nor.txt 0.2 0.01 0.1 100
	python3 mainA.py ../data/xor.txt 0.2 0.05 0.1 100