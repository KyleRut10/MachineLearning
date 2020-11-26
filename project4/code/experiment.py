import data
import ga
import de
import pso

# read in datasets
# classification
bc = data.data_breast()
gl = data.data_glass()
sb = data.data_soybean_small()

# regression
ff = data.data_forestfire()
hw = data.data_hardware()
aba = data.data_abalone()

# genetic algorithm
def run_ga_cv(df, hl, mode, pc, pm, num_chrom, tsk, max_generations):
    validation_error = []
    for cv in data.get_cross_validate_dfs(df):
        # testing, training
        test = cv[0]
        train = cv[1]

        ga = ga.GA(hl, mode, train)
        ga.train(pc, pm, num_chrom, tsk, max_generations)


# run ga bc
#run_ga_cv(bc, [], 'c', .9, .3, 10, 2, 2000)

# Genetic Algorithms, 0 hidden layers
def ga0hl():
    print('0 Hidden Layers')
    print('Breast Cancer')
    gab = ga.GA([], 'c', bc)
    gab.train(.9, .2, 10, 2, 2000)
    gab.save_network('../networks/gab.pkl')

    print('Glass')
    gag = ga.GA([], 'c', gl)
    gag.train(.9, .2, 10, 2, 2000)
    gag.save_network('../networks/gag.pkl')

    print('Soybean Small')
    gas = ga.GA([], 'c', sb)
    gas.train(.9, .2, 10, 2, 2000)
    gas.save_network('../networks/gas.pkl')

    print('Forest Fire')
    gaf = ga.GA([], 'r', ff)
    gaf.train(.9, .2, 10, 2, 2000)
    gaf.save_network('../networks/gas.pkl')

    print('Hardware')
    gah = ga.GA([], 'r', hw)
    gah.train(.9, .2, 10, 2, 2000)
    gah.save_network('../networks/gah.pkl')

    print('Abalone')
    gaa = ga.GA([], 'r', aba)
    gaa.train(.9, .2, 10, 2, 2000)
    gaa.save_network('../networks/gaa.pkl')



# Genetic Algorithms, 1 hidden layers
def ga1hl():
    print('1 Hidden Layers')
    print('Breast Cancer')
    gab = ga.GA([61], 'c', bc)
    gab.train(.9, .2, 10, 2, 2000)
    gab.save_network('../networks/gab1.pkl')

    print('Glass')
    gag = ga.GA([13], 'c', gl)
    gag.train(.9, .2, 10, 2, 2000)
    gag.save_network('../networks/gag1.pkl')

    print('Soybean Small')
    gas = ga.GA([67], 'c', sb)
    gas.train(.9, .2, 10, 2, 2000)
    gas.save_network('../networks/gas1.pkl')

    print('Forest Fire')
    gaf = ga.GA([42], 'r', ff)
    gaf.train(.9, .2, 10, 2, 2000)
    gaf.save_network('../networks/gaf1.pkl')

    print('Hardware')
    gah = ga.GA([250], 'r', hw)
    gah.train(.9, .2, 10, 2, 2000)
    gah.save_network('../networks/gah1.pkl')

    print('Abalone')
    gaa = ga.GA([13], 'r', aba)
    gaa.train(.9, .2, 10, 2, 2000)
    gaa.save_network('../networks/gaa1.pkl')

# Genetic Algorithms, 2 hidden layers
def ga2hl():
    print('2 Hidden Layers')
    print('Breast Cancer')
    gab = ga.GA([71,77], 'c', bc)
    gab.train(.9, .2, 10, 2, 2000)
    gab.save_network('../networks/gab1.pkl')

    print('Glass')
    gag = ga.GA([7,11], 'c', gl)
    gag.train(.9, .2, 10, 2, 2000)
    gag.save_network('../networks/gag1.pkl')

    print('Soybean Small')
    gas = ga.GA([33,33], 'c', sb)
    gas.train(.9, .2, 10, 2, 2000)
    gas.save_network('../networks/gas1.pkl')

    print('Forest Fire')
    gaf = ga.GA([22,39], 'r', ff)
    gaf.train(.9, .2, 10, 2, 2000)
    gaf.save_network('../networks/gaf1.pkl')

    print('Hardware')
    gah = ga.GA([229,250], 'r', hw)
    gah.train(.9, .2, 10, 2, 2000)
    gah.save_network('../networks/gah1.pkl')

    print('Abalone')
    gaa = ga.GA([4,15], 'r', aba)
    gaa.train(.9, .2, 10, 2, 2000)
    gaa.save_network('../networks/gaa1.pkl')


def de0hl():
    print('0 Hidden Layers')
    print('Breast Cancer')
    deb = de.DE([], 'c', bc)
    deb.train(1, .5, 10, 2000)
    deb.save_network('../networks/deb.pkl')

    print('Glass')
    deg = de.DE([], 'c', gl)
    deg.train(1, .5, 10, 2000)
    deg.save_network('../networks/deg.pkl')

    print('Soybean Small')
    gas = de.DE([], 'c', sb)
    gas.train(1, .5, 10, 2000)
    gas.save_network('../networks/des.pkl')

    print('Forest Fire')
    gaf = de.DE([], 'r', ff)
    gaf.train(1, .5, 10, 2000)
    gaf.save_network('../networks/des.pkl')

    print('Hardware')
    gah = de.DE([], 'r', hw)
    gah.train(1, .5, 10, 2000)
    gah.save_network('../networks/deh.pkl')

    print('Abalone')
    gaa = de.DE([], 'r', aba)
    gaa.train(1, .5, 10, 2000)
    gaa.save_network('../networks/dea.pkl')



# Genetic Algorithms, 1 hidden layers
def de1hl():
    print('1 Hidden Layers')
    print('Breast Cancer')
    gab = de.DE([61], 'c', bc)
    gab.train(1, .5, 10, 2000)
    gab.save_network('../networks/deb1.pkl')

    print('Glass')
    gag = de.DE([13], 'c', gl)
    gag.train(1, .5, 10, 2000)
    gag.save_network('../networks/deg1.pkl')

    print('Soybean Small')
    gas = de.DE([67], 'c', sb)
    gas.train(1, .5, 10, 2000)
    gas.save_network('../networks/des1.pkl')

    print('Forest Fire')
    gaf = de.DE([42], 'r', ff)
    gaf.train(1, .5, 10, 2000)
    gaf.save_network('../networks/def1.pkl')

    print('Hardware')
    gah = de.DE([250], 'r', hw)
    gah.train(1, .5, 10, 2000)
    gah.save_network('../networks/deh1.pkl')

    print('Abalone')
    gaa = de.DE([13], 'r', aba)
    gaa.train(1, .5, 10, 2000)
    gaa.save_network('../networks/dea1.pkl')

# Genetic Algorithms, 2 hidden layers
def de2hl():
    print('2 Hidden Layers')
    print('Breast Cancer')
    gab = de.DE([71,77], 'c', bc)
    gab.train(1, .5, 10, 2000)
    gab.save_network('../networks/deb1.pkl')

    print('Glass')
    gag = de.DE([7,11], 'c', gl)
    gag.train(1, .5, 10, 2000)
    gag.save_network('../networks/deg1.pkl')

    print('Soybean Small')
    gas = de.DE([33,33], 'c', sb)
    gas.train(1, .5, 10, 2000)
    gas.save_network('../networks/des1.pkl')

    print('Forest Fire')
    gaf = de.DE([22,39], 'r', ff)
    gaf.train(1, .5, 10, 2000)
    gaf.save_network('../networks/def1.pkl')

    print('Hardware')
    gah = de.DE([229,250], 'r', hw)
    gah.train(1, .5, 10, 2000)
    gah.save_network('../networks/deh1.pkl')

    print('Abalone')
    gaa = de.DE([4,15], 'r', aba)
    gaa.train(1, .5, 10, 2000)
    gaa.save_network('../networks/dea1.pkl')


def pso0hl():
    print('0 Hidden Layers')
    #print('Breast Cancer')
    #psob = pso.PSO([], 'c', bc)
    #psob.train(15, 1, 0.2, 4, 4, 100)
    #psob.save_network('../networks/psob.pkl')

    print('Glass')
    psog = pso.PSO([], 'c', gl)
    psog.train(15, 1, 0.2, 4, 4, 300)
    psog.save_network('../networks/psog.pkl')

    #print('Soybean Small')
    #psos = pso.PSO([], 'c', sb)
    #psos.train(15, 1, 0.2, 4, 4, 50)
    #psos.save_network('../networks/psos.pkl')

    #print('Forest Fire')
    #psof = pso.PSO([], 'r', ff)
    #psof.train(15, 1, 0.2, 4, 4, 50)
    #psof.save_network('../networks/psof.pkl')

    #print('Hardware')
    #psoh = pso.PSO([], 'r', ff)
    #psoh.train(15, 1, 0.2, 4, 4, 100)
    #psoh.save_network('../networks/psoh.pkl')

    #print('Abalone')
    #psoa = pso.PSO([], 'r', ff)
    #psoa.train(15, 1, 0.2, 4, 4, 100)
    #psoa.save_network('../networks/psoa.pkl')


def pso1hl():
    print('1 Hidden Layer')
    #print('Breast Cancer')
    #psob = pso.PSO([61], 'c', bc)
    #psob.train(15, 1, 0.2, 4, 4, 100)
    #psob.save_network('../networks/psob.pkl')

    #print('Glass')
    #psog = pso.PSO([13], 'c', gl)
    #psog.train(15, 1, 0.2, 4, 4, 200)
    #psog.save_network('../networks/psog.pkl')

    print('Soybean Small')
    psos = pso.PSO([67], 'c', sb)
    psos.train(15, 1, 0.2, 4, 4, 200)
    psos.save_network('../networks/psos.pkl')

    #print('Forest Fire')
    #psof = pso.PSO([42], 'r', ff)
    #psof.train(15, 1, 0.2, 4, 4, 50)
    #psof.save_network('../networks/psof.pkl')

    print('Hardware')
    psoh = pso.PSO([250], 'r', ff)
    psoh.train(15, 1, 0.2, 4, 4, 100)
    psoh.save_network('../networks/psoh.pkl')

    print('Abalone')
    psoa = pso.PSO([13], 'r', ff)
    psoa.train(15, 1, 0.2, 4, 4, 100)
    psoa.save_network('../networks/psoa.pkl')


def pso2hl():
    print('2 Hidden Layers')
    print('Breast Cancer')
    psob = pso.PSO([71,77], 'c', bc)
    psob.train(15, 1, 0.2, 4, 4, 100)
    psob.save_network('../networks/psob.pkl')

    print('Glass')
    psog = pso.PSO([7,11], 'c', gl)
    psog.train(15, 1, 0.2, 4, 4, 200)
    psog.save_network('../networks/psog.pkl')

    print('Soybean Small')
    psos = pso.PSO([33,33], 'c', sb)
    psos.train(15, 1, 0.2, 4, 4, 200)
    psos.save_network('../networks/psos.pkl')

    print('Forest Fire')
    psof = pso.PSO([22,39], 'r', ff)
    psof.train(15, 1, 0.2, 4, 4, 50)
    psof.save_network('../networks/psof.pkl')

    print('Hardware')
    psoh = pso.PSO([229,250], 'r', ff)
    psoh.train(15, 1, 0.2, 4, 4, 50)
    psoh.save_network('../networks/psoh.pkl')

    print('Abalone')
    psoa = pso.PSO([4,15], 'r', ff)
    psoa.train(15, 1, 0.2, 4, 4, 100)
    psoa.save_network('../networks/psoa.pkl')
