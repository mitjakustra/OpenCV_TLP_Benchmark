import cv2 as cv
import sys
import numpy as np

'''FUNKCIONALNOST PROGRAMA:
    -program najprej zahteva ime sekvence kot vhodni podatek
    -program inicializira sezname posameznih metrik
    -program za posamezno sekvenco in za vsak sledilni algoritem uvozi (ime_sledilca)_test_resutl.txt datoteko in
    groundtruth_rect.txt datoteko ter ustvari (ime_sledilca)_score.txt datoteko, kamor shranjuje rezultate izracunov
    -program definira funkcije, s pomocjo katerih racuna vrednosti metrik in ocen uspesnosti sled. alg.
    -program prebere vrstico po vrstico v obeh vhodnih datotekah, kar sovpada s posamezno slicio sekvence
    -program izracuna vrednosti metrik in ocen ter jih doda v korepondencne sezname
    -program zapise izracunane vrednosti metrik in ocen v izhodno datoteko v pravilnem formatu
'''

'''_____________________Začetni vnos__________________'''


print("Izberite zaporedno stevilko sekvence:")
print("1 - Alladin")
print("2 - Aquarium2")
print("3 - Badminton1")
print("4 - CarChase3")
print("5 - DriftCar1")
print("6 - ISS")
print("7 - Jet4")
print("8 - KinBall2")
print("9 - PolarBear1\n")

seq_name = input("Vnesite stevilko izbrane sekvence:")

if seq_name == '1':
    seq_name = "Alladin"
    print("Izbrana je sekvenca: Alladin")
elif seq_name == '2':
    seq_name = "Aquarium2"
    print("Izbrana je sekvenca: Aquarium2")
elif seq_name == '3':
    seq_name = "Badminton1"
    print("Izbrana je sekvenca: Badminton1")
elif seq_name == '4':
    seq_name = "CarChase3"
    print("Izbrana je sekvenca: CarChase3")
elif seq_name == '5':
    seq_name = "DriftCar1"
    print("Izbrana je sekvenca: DriftCar1")
elif seq_name == '6':
    seq_name = "ISS"
    print("Izbrana je sekvenca: ISS")
elif seq_name == '7':
    seq_name = "Jet4"
    print("Izbrana je sekvenca: Jet4")
elif seq_name == '8':
    seq_name = "KinBall2"
    print("Izbrana je sekvenca: KinBall2")
elif seq_name == '9':
    seq_name = "PolarBear1"
    print("Izbrana je sekvenca: PolarBear1")

tr_list = ["BOOSTING", "MIL", "MEDIANFLOW", "TLD", "KCF", "MOSSE", "CSRT"]


for i in tr_list:
    if __name__ == '__main__':

        # '''________________Deklaracija spremenljivk_______________'''

        Count_iou = []  # Prazen list za dodajanje IoU
        Count_success = []  # Prazen list za dodajanje števila successful IoU

        Count_distance = []
        Count_precision = []  # Prazen list za dodajanje razdalj za precision metric

        Count_LSM = []  # Prazen list za izračun LSM

        tr_input_type = i

        path_result = "./data/" + seq_name + "/" + tr_input_type + "_test_result.txt"
        path_gt = "./data/" + seq_name + "/groundtruth_rect.txt"
        path_score = "./data/" + seq_name + "/" + tr_input_type + "_score.txt"

        # '''___________________Odpiranje datotek_______________'''

        # Odpiranje datoteke Ground_truth

        try:
            gt = open(path_gt, 'r')

        except IOError:
            print("Could not open GROUNDTRUTH file!")

            # Odpiranje datoteke test_result

        try:
            test = open(path_result, 'r')

        except IOError:
            print("Could not open TESTRESULT file!")

            # Odpiranje izhodne datoteke

        try:
            output_metrices = open(path_score, 'w')

        except IOError:
            print("Could not open SCORE file!")


        # '''______________________Definicije funkcij_______________________'''

        def bb_intersection_over_union(X_GT, Y_GT, X_t, Y_t, H_GT, W_GT, H_t, W_t):
            # Izračun diagonalnih koordinat pravokotnikov
            X_GT2 = X_GT + W_GT
            Y_GT2 = Y_GT + H_GT

            X_t2 = X_t + W_t
            Y_t2 = Y_t + H_t

            # dolocanje (x, y) koordinat presecnega pravokotnika
            # xA = max(boxA[0], boxB[0])
            # yA = max(boxA[1], boxB[1])
            # xB = min(boxA[2], boxB[2])
            # yB = min(boxA[3], boxB[3])
            xA = max(X_GT, X_t)
            yA = max(Y_GT, Y_t)
            xB = min(X_GT2, X_t2)
            yB = min(Y_GT2, Y_t2)

            # izracun povrsine presecnega pravokotnika
            interArea = max(0, xB - xA) * max(0, yB - yA)

            # izracun povrsine referencnega in detektiranega pravokotnika
            # boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            # boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
            boxAArea = H_GT * W_GT
            boxBArea = H_t * W_t

            # izracun kolicnika preseka z unijo, t.i. IoU (Intersection over Union)
            iou = interArea / float(boxAArea + boxBArea - interArea)
            # vrnitev vrednosti kolicnika IoU

            return iou


        def success_threshold(iou_list, threshold):  # Izločimo tiste vrednosti, ki so nad thresholdom

            # Odštejemo threshold od vseh vrednosti
            iou_list = iou_list - threshold

            # Preštejemo pozitivna števila v array-u
            success_frame_num = iou_list[iou_list > 0]
            success_num = len(success_frame_num)

            return success_num


        def precision_distance(X_GT, Y_GT, X_t, Y_t, H_GT, W_GT, H_t, W_t):
            # tu morda ne bi bilo slabo racunati razdalje med centri bounding boxov, kot pa med oglisci...
            # GT_center:
            X_c_GT = X_GT + W_GT / 2
            Y_c_GT = Y_GT + H_GT / 2

            # t_center:
            X_c_t = X_t + W_t / 2
            Y_c_t = Y_t + H_t / 2

            pdistance = np.sqrt((X_c_GT - X_c_t) ** 2 + (Y_c_GT - Y_c_t) ** 2)

            return pdistance


        def precision_threshold(precision_list, threshold):

            # Odštejemo threshold vsem vrednostim
            precision_list = precision_list - threshold

            # Prestejemo vse vrednosti, ki so negativne (manjsi distance deviation od dovoljene)
            precision_under_thres = precision_list[precision_list < 0]
            precision_num = len(precision_under_thres)
            precision_percentage = precision_num / 600.0

            return precision_percentage


        def LSM(iou_list, threshold):
            # Določitev dolžine videa
            VidLen = 600.  # Vse sekvence imajo enako število slik - 600
            LSM_list = []
            count_lsm = 0
            count_lsm_0_5 = 0
            for i in range(0, len(iou_list)):  # Za vsak frame

                count_lsm = count_lsm + 1  # Beležimo položaj trenutnega frame-a
                if iou_list[i] > 0.5:  # Če je prekrivanje zadovoljivo (50% standardni kriterij za LSM)

                    count_lsm_0_5 = count_lsm_0_5 + 1

                # izracun deleza ustreznih frame-ov (nad thres.) za kriterij %x (naslednja if zanka)
                curr_perc = count_lsm_0_5 / count_lsm

                if curr_perc < threshold:  # če pade trenutni % uspešnih pod threshold, je subsequence končan, začni na novo
                    LSM_list.append(count_lsm)
                    count_lsm = 0
                    count_lsm_0_5 = 0

                if i == 599 and len(LSM_list) == 0:
                    # Če for zanka prehodi vse frame-e, ne da bi %x padel pod thres.
                    LSM_list.append(600.)

            # LSM_list = np.array(LSM_list)  # Konverzija v np.array
            LSM_list = np.array(LSM_list)
            LS = np.max(LSM_list)

            # Izračun deleža najdaljše sub-sequence v primerjavi s celotnim VidLen
            LSM = LS / VidLen

            return LSM


        cnt = 0
        # while cnt < 3:
        while True:
            # cnt += 1
            # Branje vrstic GT in izločitev spremenljivk
            line_gt = gt.readline()
            line_test = test.readline()

            if not line_gt or not line_test:
                break

            line_gt = line_gt.split(',')
            line_test = line_test.split(' ')

            FrameID_GT = int(line_gt[0])
            X_GT = int(line_gt[1])
            Y_GT = int(line_gt[2])
            H_GT = int(line_gt[3])
            W_GT = int(line_gt[4])
            L_GT = int(line_gt[5])
            # print(FrameID_GT, X_GT, Y_GT, H_GT, W_GT, L_GT)
            # Branje vrstic test in izločitev spremenljivk

            FrameID_t = int(line_test[0])
            X_t = int(line_test[1])
            Y_t = int(line_test[2])
            H_t = int(line_test[3])
            W_t = int(line_test[4])
            L_t = int(line_test[5])
            # print(FrameID_t, X_t, Y_t, H_t, W_t, L_t)

            # '''___________________Success, LSM metric_________________'''

            if (L_t == 1 and L_GT == 1):  # Out-of-view uspešno zaznan
                Count_iou.append(1)

            elif (L_t == 0 and L_GT == 1):  # Out-of-view neuspešno zaznan
                Count_iou.append(0)

            elif (L_t == 0 and L_GT == 0):  # Tracking uspešen
                # Izračun IoU
                IoU = bb_intersection_over_union(X_GT, Y_GT, X_t, Y_t, H_GT, W_GT, H_t, W_t)
                Count_iou.append(IoU)
            else:  # Primer ko je L_t == 1 and L_GT == 0 --> false positive --> napaka --> append(0)
                Count_iou.append(0)

            # '''_________________Precision metric_______________________'''

            prec_distance = precision_distance(X_GT, Y_GT, X_t, Y_t, H_GT, W_GT, H_t, W_t)
            Count_distance.append(prec_distance)

        # Zapremo datoteke
        gt.close()
        test.close()

        # '''_________________Success izračun________________________'''

        Count_iou = np.array(Count_iou)  # Pretvorimo list v np.array

        thr_s_array = np.arange(0, 1.01, 0.01)
        for threshold_s in thr_s_array:
            success = success_threshold(Count_iou, threshold_s)  # stevilo frame-ov z vecjim IoU kot thres.
            Count_success.append(success)  # Dodamo v list
            if threshold_s == 0.5:
                success_score_0_5 = success  # Konvencionalen success score '''<------SCORE'''
        Count_success = np.array(Count_success)  # Pretvorimo v numpy array '''<------ARRAY'''
        # Count_success bi moral biti array 20 INT stevil (padajocih)

        # Success score:
        success_score_int = np.trapz(Count_success, thr_s_array)  # '''<------SCORE'''

        # '''________________Precision izračun___________________'''

        Count_distance = np.array(Count_distance)  # array enake dolzine kot sekvenca, ker se distance deviation...
        # ... izracuna za vsak frame

        thr_p_array = np.arange(0, 50.5, 0.5)  # mejni dovoljeni odmiki v pikslih
        for threshold_p in thr_p_array:
            precision = precision_threshold(Count_distance, threshold_p)  # shrani stevilo framov v sekvenci, katerih...
            # ...distance deviation je manjsi od thres.
            Count_precision.append(precision)
            if threshold_p == 20:
                precision_score = precision  # Konvencionalen precision score '''<------SCORE'''
        Count_precision = np.array(Count_precision)  # '''<------ARRAY'''
        # za 20 thres. bi moral tudi Count_precision imeti 20 INT elementov (narascajocih)

        # '''____________________LSM izračun_____________________'''

        thr_lsm_array = np.arange(0, 1.01, 0.01)
        for threshold_lsm in thr_lsm_array:
            lsm = LSM(Count_iou, threshold_lsm)  # delez najdaljse sub-sekvence glede na thres.
            Count_LSM.append(lsm)
            if round(threshold_lsm,
                     3) == 0.95:  # potreba po round(), ker je prihajalo do num. napak; npr. 0.950000001 != 0.95
                LSM_score = lsm  # Konvencionalen LSM score '''<------SCORE'''
        Count_LSM = np.array(Count_LSM)  # '''<------ARRAY'''
        # za 20 razlicnih thres. vrednosti bi moral Count_LSM imeti 20 FLOAT elementov (padajocih po vrednosti)

    # '''__________________Zapis v (tracker_type)_score.txt__________________'''
    # Format of each line:
    #   Count_success
    #   Count_precision
    #   Count_LSM
    #   success_score_0.5
    #   success_score_int (-egrated)
    #   precission_score
    #   LSM_score


    output_metrices.write("\n".join(" ".join(map(str, x)) for x in (Count_success, Count_precision, Count_LSM)))
    output_metrices.write('\n')
    output_metrices.write(str(round(success_score_0_5)))
    output_metrices.write('\n')
    output_metrices.write(str(int(round(success_score_int, 6))))
    output_metrices.write('\n')
    output_metrices.write(str(round(precision_score, 5)))
    output_metrices.write('\n')
    output_metrices.write(str(round(LSM_score, 5)))

    output_metrices.close()

    print("Benchmark complete!")
