import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt


'''FUNKCIONALNOST PROGRAMA:
    -program najprej uvozi (ime_sledilca)_score.txt datoteke za vse algoritme in vse sekvence
    -program prebere podatke iz uvozenih datotek in jih razvrsti v sezname po posameznih algoritmih in metrikah vrednotenja
    -program izracuna povprecne vrednosti na podlagi vseh sekvenc pri posameznem algoritmu sledenja
    .program izriše izracunane povprecne vrednosti rezultatov vrednotenja za posamezne metrike, nato se za posamezne
    ocene metrik in na koncu se uspesnost po posameznih motilnih faktorjih sledenja
'''

ptNum = 101

if __name__ == '__main__':

    '''_____________________Deklaracija array-ev_________________'''
    '''success_BOOSTING = np.array([])
    precision_BOOSTING = np.array([])
    LSM_BOOSTING = np.array([])

    success_MIL = np.array([])
    precision_MIL = np.array([])
    LSM_MIL = np.array([])

    success_KCF = np.array([])
    precision_KCF = np.array([])
    LSM_KCF = np.array([])

    success_TLD = np.array([])
    precision_TLD = np.array([])
    LSM_TLD = np.array([])

    success_MEDIANFLOW = np.array([])
    precision_MEDIANFLOW = np.array([])
    LSM_MEDIANFLOW = np.array([])

    success_MOSSE = np.array([])
    precision_MOSSE = np.array([])
    LSM_MOSSE = np.array([])

    success_CSRT = np.array([])
    precision_CSRT = np.array([])
    LSM_CSRT = np.array([])'''

    success = []
    precision = []
    LSM = []

    success_05 = []
    success_int = []
    precision_s = []
    LSM_s = []

    trackers_dict = {
        1: "BOOSTING",
        2: "MIL",
        3: "KCF",
        4: "TLD",
        5: "MEDIANFLOW",
        6: "MOSSE",
        7: "CSRT"
    }

    videos_dict = {
        1: "Alladin",
        2: "Aquarium2",
        3: "Badminton1",
        4: "CarChase3",
        5: "DriftCar1",
        6: "ISS",
        7: "Jet4",
        8: "KinBall2",
        9: "PolarBear1"
    }

    vid_index = 9
    tracker_index = 7

    for tracker in range(1, tracker_index + 1):
        success_cur = np.array([])
        precision_cur = np.array([])
        LSM_cur = np.array([])

        success_score_0_5 = np.array([])
        success_score_int = np.array([])
        precision_score = np.array([])
        LSM_score = np.array([])

        for video in range(1, vid_index + 1):

            '''___________________Odpiranje datotek_______________'''

            try:
                output_metrices = open("./data/" + videos_dict[video] + "/" + trackers_dict[tracker] + "_score.txt",
                                       'r')

            except IOError:
                print("Could not open score file!")
                sys.exit()

            '''_________________Branje benchmark datotek___________'''

            # Izluščimo Success, Precision in LSM za krivulje za posamezen video izbranega trackerja
            success_cur = np.hstack((success_cur, np.loadtxt(output_metrices, max_rows=1)))
            precision_cur = np.hstack((precision_cur, np.loadtxt(output_metrices, skiprows=0, max_rows=1)))
            LSM_cur = np.hstack((LSM_cur, np.loadtxt(output_metrices, skiprows=0, max_rows=1)))

            # Izluščimo ocene (score) za posamezen video izbranega trackerja
            success_score_0_5_cur, success_score_int_cur, precision_score_cur, LSM_score_cur = np.loadtxt(
                output_metrices, delimiter=' ', skiprows=0)

            # Ocene dodamo v array, ki na koncu vsebuje ocene ISTEGA trackerja za vse videje
            success_score_0_5 = np.hstack((success_score_0_5, success_score_0_5_cur))
            success_score_int = np.hstack((success_score_int, success_score_int_cur))
            precision_score = np.hstack((precision_score, precision_score_cur))
            LSM_score = np.hstack((LSM_score, LSM_score_cur))

        '''_______Izračun povprečja______'''

        # Povprečje za krivulje za posamezen tracker (vsak tracker ima 21 vrednosti, nizani eden za drugim)

        for i in range(0, ptNum):
            success.append(np.mean(success_cur[i::ptNum]))

        for i in range(0, ptNum):
            precision.append(np.mean(precision_cur[i::ptNum]))

        for i in range(0, ptNum):
            LSM.append(np.mean(LSM_cur[i::ptNum]))

        # Povprečje za SCORE

        success_05.append(np.mean(success_score_0_5))
        success_int.append(np.mean(success_score_int))
        precision_s.append(np.mean(precision_score))
        LSM_s.append(np.mean(LSM_score))

    success = np.array(success) / 600  # Normaliziramo
    precision = np.array(precision)
    LSM = np.array(LSM)

    success_05 = np.array(success_05) / 6
    success_int = np.array(success_int) / 6
    precision_s = np.array(precision_s) * 100
    LSM_s = np.array(LSM_s) / 6

    '''___________Izris SUCCESS plota vseh trackerjev__________'''
    success_y = np.arange(0, 1.01, 0.01)

    plt.figure()
    # BOOSTING
    plt.plot(success_y, success[0:ptNum] * 100, '-b', label='BOOSTING')

    # MIL
    plt.plot(success_y, success[ptNum:2 * ptNum] * 100, '-r', label='MIL')

    # KCF
    plt.plot(success_y, success[2 * ptNum:3 * ptNum] * 100, 'g-', label='KCF')

    # TLD
    plt.plot(success_y, success[3 * ptNum:4 * ptNum] * 100, 'c-', label='TLD')

    # MEDIANFLOW
    plt.plot(success_y, success[4 * ptNum:5 * ptNum] * 100, 'm-', label='MEDIANFLOW')

    # MOSSE
    plt.plot(success_y, success[5 * ptNum:6 * ptNum] * 100, 'y-', label='MOSSE')

    # CSRT
    plt.plot(success_y, success[6 * ptNum:7 * ptNum] * 100, 'k-', label='CSRT')
    plt.title("Diagram uspešnosti")
    plt.ylabel("Stopnja uspešnosti [%]")
    plt.xlabel("Meja prekrivanja")
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 110, 10))
    plt.legend()
    plt.show()

    '''___________Izris PRECISION plota vseh trackerjev__________'''
    precision_y = np.arange(0, 50.5, 0.5)

    plt.figure()
    # BOOSTING
    plt.plot(precision_y, precision[0:ptNum] * 100, '-b', label='BOOSTING')

    # MIL
    plt.plot(precision_y, precision[ptNum:2 * ptNum] * 100, '-r', label='MIL')

    # KCF
    plt.plot(precision_y, precision[2 * ptNum:3 * ptNum] * 100, 'g-', label='KCF')

    # TLD
    plt.plot(precision_y, precision[3 * ptNum:4 * ptNum] * 100, 'c-', label='TLD')

    # MEDIANFLOW
    plt.plot(precision_y, precision[4 * ptNum:5 * ptNum] * 100, 'm-', label='MEDIANFLOW')

    # MOSSE
    plt.plot(precision_y, precision[5 * ptNum:6 * ptNum] * 100, 'y-', label='MOSSE')

    # CSRT
    plt.plot(precision_y, precision[6 * ptNum:7 * ptNum] * 100, 'k-', label='CSRT')
    plt.title("Diagram natančnosti")
    plt.ylabel("Stopnja natančnosti [%]")
    plt.xlabel("Mejna evklidska razdalja")
    plt.xticks(np.arange(0, 55, 5))
    plt.yticks(np.arange(0, 110, 10))
    plt.legend()
    plt.show()

    '''___________Izris LSM plota vseh trackerjev__________'''
    LSM_y = np.arange(0, 1.01, 0.01)

    plt.figure()
    # BOOSTING
    plt.plot(LSM_y, LSM[0:ptNum] * 100, '-b', label='BOOSTING')

    # MIL
    plt.plot(LSM_y, LSM[ptNum:2 * ptNum] * 100, '-r', label='MIL')

    # KCF
    plt.plot(LSM_y, LSM[2 * ptNum:3 * ptNum] * 100, 'g-', label='KCF')

    # TLD
    plt.plot(LSM_y, LSM[3 * ptNum:4 * ptNum] * 100, 'c-', label='TLD')

    # MEDIANFLOW
    plt.plot(LSM_y, LSM[4 * ptNum:5 * ptNum] * 100, 'm-', label='MEDIANFLOW')

    # MOSSE
    plt.plot(LSM_y, LSM[5 * ptNum:6 * ptNum] * 100, 'y-', label='MOSSE')

    # CSRT
    plt.plot(LSM_y, LSM[6 * ptNum:7 * ptNum] * 100, 'k-', label='CSRT')
    plt.title("LSM diagram")
    plt.ylabel("LSM [%]")
    plt.xlabel("% sličic (x) z IoU > 0.5")
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 110, 10))
    plt.legend()
    plt.show()

    '''________________________SUCCESS 0.5 SCORE (pri IoU > 0.5)_____________'''

    plt.figure()

    tracks = ('BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT')
    plt.bar(np.arange(1, tracker_index + 1),
            success_05)  # , color = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black'])
    plt.xticks(np.arange(1, tracker_index + 1), tracks)
    plt.title("Ocena uspešnosti (pri 0.5 IoU)")
    plt.ylim(0, 50)
    plt.ylabel("Ocena uspešnosti [%]")
    plt.show()

    '''________________________SUCCESS INTEGRATION SCORE (površina pod krivuljo)_____________'''

    plt.figure()

    tracks = ('BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT')
    plt.bar(np.arange(1, tracker_index + 1),
            success_int)  # , color = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black'])
    plt.xticks(np.arange(1, tracker_index + 1), tracks)
    plt.title("Integrirana ocena uspešnosti")
    plt.ylim(0, 50)
    plt.ylabel("Ocena uspešnosti [%]")
    plt.show()

    '''________________________PRECISION SCORE (pri razdalji manjsi od 20 pikslov)_____________'''

    plt.figure()

    tracks = ('BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT')
    plt.bar(np.arange(1, tracker_index + 1),
            precision_s)  # , color = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black'])
    plt.xticks(np.arange(1, tracker_index + 1), tracks)
    plt.title("Ocena natančnosti")
    plt.ylim(0, 40)
    plt.ylabel("Ocena natančnosti [%]")
    plt.show()

    '''________________________LSM SCORE (pri 95% slik z IoU > 0.5 v podsekvenci)_____________'''

    plt.figure()

    tracks = ('BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT')
    plt.bar(np.arange(1, tracker_index + 1),
            success_int)  # , color = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black'])
    plt.xticks(np.arange(1, tracker_index + 1), tracks)
    plt.title("LSM ocena")
    plt.ylabel("LSM ocena [%]")
    plt.ylim(0, 50)
    plt.show()

    '''________________________Izris SCORE glede na faktor (video)____________________________'''

    score_Alladin = np.array([170, 262, 273, 36, 259, 347, 249])
    score_Aquarium2 = np.array([42, 35, 31, 14, 25, 32, 158])
    score_Badminton1 = np.array([248, 244, 251, 120, 110, 1, 533])
    score_CarChase3 = np.array([73, 73, 72, 567, 600, 76, 44])
    score_DriftCar1 = np.array([13, 22, 27, 42, 40, 25, 62])
    score_ISS = np.array([163, 172, 310, 104, 316, 171, 502])
    score_Jet4 = np.array([144, 65, 58, 24, 7, 137, 252])
    score_KinBall2 = np.array([598, 442, 349, 8, 433, 492, 600])
    score_PolarBear1 = np.array([145, 146, 119, 29, 92, 141, 118])

    print("Stopnje uspesnosti alg. sledenja pri posamezni sekvenci:")

    # Alladin
    plt.figure()
    tracks = ('BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT')
    plt.bar(np.arange(1, tracker_index + 1), score_Alladin / 6)
    plt.xticks(np.arange(1, tracker_index + 1), tracks)
    plt.title("Spremembe osvetlitve")
    plt.ylabel("Stopnja uspešnosti [%]")
    plt.ylim(0, 100)
    plt.show()

    print(np.round(score_Alladin / 6, 2))

    # Aquarium2
    plt.figure()
    tracks = ('BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT')
    plt.bar(np.arange(1, tracker_index + 1), score_Aquarium2 / 6)
    plt.xticks(np.arange(1, tracker_index + 1), tracks)
    plt.title("Nered ozadja")
    plt.ylabel("Stopnja uspešnosti [%]")
    plt.ylim(0, 100)
    plt.show()

    print(np.round(score_Aquarium2 / 6, 2))

    # Badminton1
    plt.figure()
    tracks = ('BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT')
    plt.bar(np.arange(1, tracker_index + 1), score_Badminton1 / 6)
    plt.xticks(np.arange(1, tracker_index + 1), tracks)
    plt.title("Zakrivanje")
    plt.ylabel("Stopnja uspešnosti [%]")
    plt.ylim(0, 100)
    plt.show()

    print(np.round(score_Badminton1 / 6, 2))

    # CarChase3
    plt.figure()
    tracks = ('BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT')
    plt.bar(np.arange(1, tracker_index + 1), score_CarChase3 / 6)
    plt.xticks(np.arange(1, tracker_index + 1), tracks)
    plt.title("Sprememba velikosti")
    plt.ylabel("Stopnja uspešnosti [%]")
    plt.ylim(0, 100)
    plt.show()

    print(np.round(score_CarChase3 / 6, 2))

    # DriftCar1
    plt.figure()
    tracks = ('BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT')
    plt.bar(np.arange(1, tracker_index + 1), score_DriftCar1 / 6)
    plt.xticks(np.arange(1, tracker_index + 1), tracks)
    plt.title("Sprememba perspektive")
    plt.ylabel("Stopnja uspešnosti [%]")
    plt.ylim(0, 100)
    plt.show()

    print(np.round(score_DriftCar1 / 6, 2))

    # ISS
    plt.figure()
    tracks = ('BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT')
    plt.bar(np.arange(1, tracker_index + 1), score_ISS / 6)
    plt.xticks(np.arange(1, tracker_index + 1), tracks)
    plt.title("Izven scene")
    plt.ylabel("Stopnja uspešnosti [%]")
    plt.ylim(0, 100)
    plt.show()

    print(np.round(score_ISS / 6, 2))

    # Jet4
    plt.figure()
    tracks = ('BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT')
    plt.bar(np.arange(1, tracker_index + 1), score_Jet4 / 6)
    plt.xticks(np.arange(1, tracker_index + 1), tracks)
    plt.title("Hitri premiki")
    plt.ylabel("Stopnja uspešnosti [%]")
    plt.ylim(0, 100)
    plt.show()

    print(np.round(score_Jet4 / 6, 2))

    # KinBall2
    plt.figure()
    tracks = ('BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT')
    plt.bar(np.arange(1, tracker_index + 1), score_KinBall2 / 6)
    plt.xticks(np.arange(1, tracker_index + 1), tracks)
    plt.title("Zakrivanje")
    plt.ylabel("Stopnja uspešnosti [%]")
    plt.ylim(0, 100)
    plt.show()

    print(np.round(score_KinBall2 / 6, 2))

    # PolarBear1
    plt.figure()
    tracks = ('BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT')
    plt.bar(np.arange(1, tracker_index + 1), score_PolarBear1 / 6)
    plt.xticks(np.arange(1, tracker_index + 1), tracks)
    plt.title("Zakrivanje")
    plt.ylabel("Stopnja uspešnosti [%]")
    plt.ylim(0, 100)
    plt.show()

    print(np.round(score_PolarBear1 / 6, 2))