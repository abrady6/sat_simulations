from satellite_simulations import *
import pickle
import pandas as pd


def get_data(filename,all=True,json=False,multiple_alt=False):

    # Last modified: 15 October 2019
    
    if not json:
        f=open(filename,'rb')
        if all:
            if multiple_alt:
                data,S,G,G_pair_labels,sep_distances,sep_angles,h=pickle.load(f)
                f.close()
                return data,S,G,G_pair_labels,sep_distances,sep_angles,h
            else:
                data,S,G,G_pair_labels,sep_distances,sep_angles,_=pickle.load(f)
                f.close()
                return data,S,G,G_pair_labels,sep_distances,sep_angles
        else:
            if multiple_alt:
                data,S,G,G_pair_labels,h=pickle.load(f)
                f.close()
                return data,S,G,G_pair_labels,h
            else:
                data,S,G,G_pair_labels,_=pickle.load(f)
                f.close()
                return data,S,G,G_pair_labels
    elif json:
        f1=open(filename+'.pkl','rb')
        if all:
            if multiple_alt:
                S,G,G_pair_labels,sep_distances,sep_angles,H=pickle.load(f1)
                f1.close()
            else:
                S,G,G_pair_labels,sep_distances,sep_angles,_=pickle.load(f1)
                f1.close()
        else:
            if multiple_alt:
                S,G,G_pair_labels,H=pickle.load(f1)
                f1.close()
            else:
                S,G,G_pair_labels,_=pickle.load(f1)
                f1.close()
        f2=open(filename+'.json')
        df=pd.read_json(f2)
        f2.close()
        data=df.to_dict("split")
        times=data['index']
        # Reconstruct data to conform with original structure
        data_new={}
        for i in range(len(times)):
            t=times[i]
            data_new[t]={}
            data_new[t]['ground station pairs']={}
            for pair in G_pair_labels:
                data_new[t]['ground station pairs'][str(pair)]={}
                if multiple_alt:
                    data_new[t]['ground station pairs'][str(pair)]['in range']={}
                    for h in H:
                        g_data=data['data'][i][0][str(pair)]['in range'][str(h/1000)]
                        g_data_new=[]
                        for sat in g_data:
                            g_data_new.append([tuple(sat[0])]+sat[1:])
                        data_new[t]['ground station pairs'][str(pair)]['in range'][h/1000]=g_data_new
                else:
                    g_data=data['data'][i][0][str(pair)]['in range']
                    g_data_new=[]
                    for sat in g_data:
                        g_data_new.append([tuple(sat[0])]+sat[1:])
                    data_new[t]['ground station pairs'][str(pair)]['in range']=g_data_new

        if all:
            if multiple_alt:
                return data_new,S,G,G_pair_labels,sep_distances,sep_angles,H
            else:
                return data_new,S,G,G_pair_labels,sep_distances,sep_angles
        else:
            if multiple_alt:
                return data_new,S,G,G_pair_labels,H
            else:
                return data_new,S,G,G_pair_labels



def get_range_data(data,G_pair_labels,h=None):

    # Last modified: 3 October 2019

    '''
    If multiple satellite altitudes are contained in data, then h must be 
    specified as a number in meters.
    '''

    in_range={} # Dictionary for each ground station pair, showing a list of zeros and ones (zero==not in range, 1==in range of a satellite)
    times={}  # Dictionary for each ground station pair, showing the times at which it is in range of a satellite.
    time_gaps={}  # Dictionary for each ground station pair, showing the times at which it is not in range of a satellite.
    num_sats={}  # Dictionary for each ground station pair, showing the number of satellites in the range of each ground station pair.

    for i in range(len(G_pair_labels)):
        in_range[i],times[i],time_gaps[i],num_sats[i]=range_times(data,str(G_pair_labels[i]),h=h)

    return in_range,times,time_gaps,num_sats


def get_time_gap_lengths(time_gaps):

    # Last modified: 7 August 2019

    time_gap_lengths={}

    for i in range(len(time_gaps)):
        if len(time_gaps[i])==0:
            time_gap_lengths[i]=np.array([])
        else:
            tmp=np.array([time_gaps[i][j+1]-time_gaps[i][j] for j in range(len(time_gaps[i])-1)])
            tmp2=tmp==np.ones(len(tmp))
            tmp3=np.argwhere(tmp2==False)
            tmp3=[tmp3[i][0] for i in range(len(tmp3))]
            tmp4=[tmp2[i:j] for i,j in zip([0]+tmp3,tmp3+[None])]

            time_gap_lengths[i]=np.array([len(elem) for elem in tmp4])

    return time_gap_lengths


def get_avg_losses(data,G_pair_labels,h=None):

    # Last modified: 3 October 2019

    T=len(data) # Total simulation time
    avg_trans={}  # Average transmissivity
    avg_loss={}  # Average loss
    for i in range(len(G_pair_labels)):
        avg_loss[i]=0
        avg_trans[i]=0

    for t in range(T):
        for i in range(len(G_pair_labels)):
            if h==None:
                g_range=data[t]['ground station pairs'][str(G_pair_labels[i])]['in range']
            else:
                g_range=data[t]['ground station pairs'][str(G_pair_labels[i])]['in range'][h/1000]
            if not g_range:
                avg_loss[i]+=0
                avg_trans[i]+=0
            else:
                avg_loss[i]+=min([g_range[j][4] for j in range(len(g_range))])/T
                avg_trans[i]+=max([g_range[j][3] for j in range(len(g_range))])/T

    return avg_trans,avg_loss


def get_avg_rate(data,pair,R=1e9,all_pairs=False,h=None):

    # Last modified: 3 October 2019

    '''
    Gets the time-average rate for the given pair of ground stations.
    The total time is taken to be the simulation time from data.
    The pair should be specified as a string.

    R is the repetition rate of the source (in ebits/second).

    R*eta is the average number of ebits in each second of the simulation. Because the
    ground stations and the satellites are moving, the quantity R*eta varies with time
    (because eta is a function of time). So the quantity avg_rate is the number of ebits
    obtained in the simulation divided by the simulation time.

    If all_pairs=True, then the variable pair should contain a list of pair labels.
    '''

    T=len(data) # Total simulation time

    def pair_rate(pair):

        num_pairs=[]  # The number of ebits transmitted to the ground stations for each time step
        avg_rate=0

        for t in range(T):
            if h==None:
                g_range=data[t]['ground station pairs'][pair]['in range']
            else:
                g_range=data[t]['ground station pairs'][pair]['in range'][h/1000]
            if not g_range:
                num_pairs.append(0)
                avg_rate+=0/T
            else:
                eta=max([g_range[j][3] for j in range(len(g_range))])   # Take the satellite corresponding to the maximum transmissivity
                num_pairs.append(R*eta)
                avg_rate+=R*eta/T

        return avg_rate,np.array(num_pairs)

    if not all_pairs:
        return pair_rate(pair)
    elif all_pairs==True:

        avg_rates={}

        G_pair_labels=pair

        for i in range(len(G_pair_labels)):
            avg_rates[i]=pair_rate(str(G_pair_labels[i]))
        
        return avg_rates


def extract_results(data_vars,json=False,multiple_alt=False,pkl=False):

    # Last modified: 15 October 2019

    '''
    Extracts and organizes all of the data from the filename containing 
    the raw simulation data.

    The pkl flag is for whether the results of this function should be
    dumped into a pickle file.

    The required variables are: data, S, G_pair_labels, sep_distances
    '''

    if type(data_vars)==str:  # First argument is filename containing the data, so the data needs to be extracted first
        filename=data_vars
        if multiple_alt:
            data,S,_,G_pair_labels,sep_distances,_,_=get_data(filename,json=json,multiple_alt=multiple_alt)
        else:
            data,S,_,G_pair_labels,sep_distances,_=get_data(filename,json=json,multiple_alt=multiple_alt)
    elif type(data_vars)==list:  # First argument already contains all of the needed data
        data=data_vars[0]
        S=data_vars[1]
        G_pair_labels=data_vars[2]
        sep_distances=data_vars[3]
        filename=data_vars[4]  # Filename for saving the output to a pkl file.

    R={}

    R['# rings']=len(S)
    R['# Sats. per ring']=len(S[1])-1
    R['Total # of sats.']=len(S)*(len(S[1])-1)

    R['Time of run']=len(data)

    R['Ground station separations']=sep_distances[0:-1]
    D=sep_distances[0:-1]

    if multiple_alt:
        H=list(S[1][1].keys())
        for h in H:
            R[h]={}
    else:
        H=[1]
        for h in H:
            R[h]={}

    for h in H:
        if h==1:
            _,_,time_gaps,_=get_range_data(data,G_pair_labels,h=None)
            _,avg_loss=get_avg_losses(data,G_pair_labels,h=None)
            avg_rates=get_avg_rate(data,G_pair_labels,all_pairs=True,h=None)
        else:
            _,_,time_gaps,_=get_range_data(data,G_pair_labels,h=h*1e3)
            _,avg_loss=get_avg_losses(data,G_pair_labels,h=h*1e3)
            avg_rates=get_avg_rate(data,G_pair_labels,all_pairs=True,h=h*1e3)
        
        time_gap_lengths=get_time_gap_lengths(time_gaps)

        num_time_gaps=[len(time_gap_lengths[i]) for i in range(len(time_gap_lengths))]
        len_time_gaps=[np.mean(time_gap_lengths[i]) for i in range(len(time_gap_lengths))]

        for i in range(len(D)):
            R[h]['d='+str(sep_distances[i])]=(num_time_gaps[i],len_time_gaps[i],avg_loss[i],avg_rates[i][0])

    
    '''
    Sat_assign={}  # Assigning satellites to ground stations
    Ground_assign={} # Assigning ground stations to satellites
    for t in range(len(data)):
        Sat_assign[t]={}
        Ground_assign[t]={}
        for h in H:
            _,sat_assign_dict,_,g_assign_dict=assign_ground_stations_and_satellites(data,S,G_pair_labels,t,h=h*1e3)
            Sat_assign[t][h]=sat_assign_dict
            Ground_assign[t][h]=g_assign_dict
    '''
    if pkl:
        file=filename.replace('.pkl','')+'_results.pkl'
        f=open(file,'wb')
        pickle.dump(R,f)
        f.close()
    else:
        return R


def sort_results(files,H=np.linspace(500,10000,20),D=np.linspace(100e3,5000e3,50),get_best=False):

    # Last modified: 14 October 2019

    '''
    files should be a dictionary structured as:

        (num rings, num sats): 'filename'
    '''

    all_results={}
    for file in files.keys():
        f=open(files[file],'rb')
        all_results[file]=pickle.load(f)
        f.close()

    results_d={}

    for d in D:
        results_d[d/1000]={}
        for h in H:
            results_d[d/1000][h]={}
            for config in all_results.keys():
                data=all_results[config][h]['d='+str(d)]
                if get_best:
                    if data[0]==0:
                        results_d[d/1000][h][config]=data
                    else:
                        continue
                else:
                    results_d[d/1000][h][config]=data

    return results_d,all_results



def get_sats_in_range(data,G_pair_labels,t,h=None):

    # Last modified: 9 October 2019

    '''
    Grabs all of the satellite labels that are in range of all of the ground
    station pairs in G_pair_labels at the given time t.

    (Gives the ground station point of view.)

    If data contains satellites with multiple altitudes, then the altitude h
    under consideration should be specified (in meters).
    '''

    g_stations={}  # Contains just the satellite labels
    g_stations_dict={}  # Contains the satellite labels and the distances to the satellite, and the loss

    for i in range(len(G_pair_labels)):
        if h==None:
            g_range=data[t]['ground station pairs'][str(G_pair_labels[i])]['in range']
        else:
            g_range=data[t]['ground station pairs'][str(G_pair_labels[i])]['in range'][h/1000]
        g_stations[i]=[sat[0] for sat in g_range]
        g_stations_dict[i]={}
        for sat in g_range:
            g_stations_dict[i][sat[0]]=sat[1:]

    return g_stations,g_stations_dict


def get_ground_stations_in_range(data,S,G_pair_labels,t,h=None):

    # Last modified: 9 October 2019

    '''
    Looks at the ground stations in G_pair_labels that are in the range of
    the satellites in S at the time t. Unlike the function "get_sats_in_range"
    below, which looks at the satellites in range of ground stations, this
    functions finds the "inverse" mapping.

    (Gives the satellite point of view.)

    If data contains satellites with multiple altitudes, then the altitude h
    under consideration should be specified (in meters).
    '''

    sats={}
    sats_dict={}

    g_stations,g_stations_dict=get_sats_in_range(data,G_pair_labels,t,h=h)

    I1=list(S.keys())  # First coordinate for the satellite labels
    I2=list(S[1].keys()) # Second coordinate for the satellite labels
    I2.remove('axis')

    for i in I1:
        for j in I2:
            sats[(i,j)]=[]
            sats_dict[(i,j)]=[]
            for k in range(len(G_pair_labels)):
                if (i,j) in g_stations[k]:
                    sats[(i,j)].append(k)
                    sats_dict[(i,j)].append((k,g_stations_dict[k][i,j]))

    
    return sats,sats_dict


def assign_ground_stations_and_satellites(data,S,G_pair_labels,t,h=None,sort=False):

    # Last modified: 9 October 2019

    '''
    For the given time t, we assign each satellite to a ground station pair for
    which that satellite is in range. If sort=False, and there are multiple ground
    station pairs in range of a given satellite, then we take the ground station
    pair with the least loss. If sort=True, and there are multiple ground station
    pairs in range of a given satellite, then they simply sorted by loss.
    '''

    # satellite --> ground station pair assignment
    sat_assign={}  # Contains just the ground station pair label for each satellite
    sat_assign_dict={} # Contains the distance and loss information as well
    
    # ground station --> satellite assignment
    g_assign={}
    g_assign_dict={}

    sats,sats_dict=get_ground_stations_in_range(data,S,G_pair_labels,t,h=h)
    g_stations,g_stations_dict=get_sats_in_range(data,G_pair_labels,t,h=h)

    Sats=list(sats.keys())

    # Iterate over all satellites, find the best ground station for each satellite based on lowest loss
    for sat in Sats:
        G=sats_dict[sat]
        G_labels=[g[0] for g in G]
        if not G:  # If G is empty (which means that no ground stations are within range of the satellite)
            sat_assign[sat]=[]
            sat_assign_dict[sat]=[]
        else:
            G_losses=[g[1][3] for g in G]
            if sort:
                # Labels for the ground station pairs based on sorting from lowest loss to highest loss
                G_labels_sorted=list(np.array(G_labels)[np.argsort(np.array(G_losses))])
                sat_assign[sat]=G_labels_sorted
                sat_assign_dict[sat]=[G[i] for i in range(len(G_labels_sorted))]
            else:
                min_loss=min(G_losses)
                best_ground_station=np.argwhere(np.array(G_losses)==min_loss)[0][0]
                sat_assign[sat]=G_labels[best_ground_station]
                sat_assign_dict[sat]=G[best_ground_station]

    
    # Now iterate over all ground stations, find the best satellite for each one.
    for i in range(len(G_pair_labels)):
        satellites=g_stations_dict[i]
        satellites_labels=list(satellites.keys())
        if not satellites:  # if satellites is empty (meaning that no satellite are in range of the ground station)
            g_assign[i]=[]
            g_assign_dict[i]=[]
        else:
            sat_losses=[satellites[sat][3] for sat in satellites_labels]
            if sort:
                # Labels for the satellites based on sorting from lowest loss to highest loss
                sorted_indices=list(np.argsort(np.array(sat_losses)))
                satellites_labels_sorted=[satellites_labels[i] for i in sorted_indices]
                g_assign[i]=satellites_labels_sorted
                g_assign_dict[i]=[(sat,satellites[sat]) for sat in satellites_labels_sorted]
            else:
                min_loss=min(sat_losses)
                best_sat=np.argwhere(np.array(sat_losses)==min_loss)[0][0]
                g_assign[i]=satellites_labels[best_sat]
                g_assign_dict[i]=(satellites_labels[best_sat],satellites[satellites_labels[best_sat]])

    return sat_assign,sat_assign_dict,g_assign,g_assign_dict




def filter_satellite_ground_assignments(data,S,G_pair_labels,t,h=None,sort=True):

    # Last modified: 9 October 2019

    '''
    Takes the assignments from the function "assign_ground_stations_and_satellites"
    and filters them so that, at the time t, there is exactly one satellite
    assigned to a ground station pair (and vice versa).

    If data contains satellites with multiple altitudes, then the altitude h
    under consideration should be specified (in meters).
    '''

    sat_assign,sat_assign_dict,g_assign,g_assign_dict=assign_ground_stations_and_satellites(data,S,G_pair_labels,t,h=h,sort=sort)

    assignment={}  # Dictionary structured as "ground pair number : satellite".

    #all_assigned=False

    all_ground_pairs=[i for i in range(len(G_pair_labels))]
    all_sats=[(i,j) for i in S.keys() for j in S[1].keys() if j!='axis']

    available_sats=[]
    available_sats2=[]
    for sat in sat_assign.keys():
        if not sat_assign[sat]:
            continue
        else:
            available_sats.append(sat)
            available_sats2.append(sat)

    #while not all_assigned:


    ############################################################################
    # STEP 1
    ############################################################################
    # First make assignments for all ground station pairs that have only one
    # satellite in range (called "lone stations"). This way, we ensure that
    # as many ground station pairs as possible obtain entanglement, even if 
    # the loss is high.
    #
    # We must also check that all lone ground station pairs that get assigned in
    # this way have a unique satellite (i.e., it is possible for two lone
    # ground station pairs to be within range of the SAME satellite). In this case,
    # the ground station pair with the lowest loss "wins". The losing ground
    # station pair(s) does(do) not get a satellite.
    ############################################################################


    ### Extract the ground station pairs with just one satellite, along with the satellite
    lone_stations=[]
    sats_tmp=[]
    for i in all_ground_pairs:
        if len(g_assign[i])==1:
            lone_stations.append((i,g_assign[i][0]))
            sats_tmp.append(g_assign[i][0])

    ### Rearrange the data obtained above so that it is of the form sat:[pairs]
    sats_tmp=list(set(sats_tmp))
    lone_stations_sats={}
    for sat in sats_tmp:
        lone_stations_sats[sat]=[]
        for elem in lone_stations:
            g=elem[0]
            if g in sat_assign[sat]:
                lone_stations_sats[sat].append(g)

    ### Make the assignments. If two (or more) lone ground pairs do not have a
    ### satellite in common, then make the assignment right away; otherwise, make
    ### the assignment based on which ground pair has the lowest loss. The
    ### remaining ground pairs do not get assigned to a satellite.
    for sat in lone_stations_sats.keys():
        if len(lone_stations_sats[sat])==1:
            g=lone_stations_sats[sat][0]
            assignment[g]=sat
            all_ground_pairs.remove(g)  # Need to remove the ground station pair because it has been assigned
            all_sats.remove(sat) # Need to remove the satellite because it has been assigned
            available_sats2.remove(sat)
        else:
            g_list=lone_stations_sats[sat]
            losses=[g_assign_dict[g][0][1][3] for g in g_list]
            best_g_loc=np.argwhere(np.array(losses)==min(losses))[0][0]
            best_g=g_list[best_g_loc]
            assignment[best_g]=sat
            g_list.remove(best_g)  # g_list now does not contain the best ground station
            all_ground_pairs.remove(best_g)  # Need to remove the ground station pair because it has been assigned
            all_sats.remove(sat) # Need to remove the satellite because it has been assigned
            available_sats2.remove(sat)
            for g in g_list:
                assignment[g]=None
                all_ground_pairs.remove(g) # Need to remove the ground station pair because it has been assigned

    
    ############################################################################
    # STEP 2
    ############################################################################
    # Make the rest of the assignments. We consider only those ground station
    # pairs and satellite that have not been assigned yet. We loop over all
    # unassigned ground station pairs. For each pair, we loop over all
    # unassigned satellites and check to see if the ground station pair is at the
    # front of any list corresponding to the satellites. If there are, then we
    # record the satellites (there could be more than one); if there are none, then
    # we move on to the next ground station pair. If there are multiple satellites
    # for which the given ground station pair is the best, then we take the one
    # with the lowest loss.
    ############################################################################
    
    for g in range(len(G_pair_labels)):
        if g not in all_ground_pairs:  # Check if the ground station pair has already been assigned. If g is not in all_ground_pairs, then it has already been assigned
            continue
        best_sats_for_g=[]
        for sat in all_sats:  # Checking only those satellites that haven't been assigned
            if not sat_assign[sat]:
                continue
            elif sat_assign[sat][0]==g:
                best_sats_for_g.append(sat)
        
        if not best_sats_for_g:  # Given ground station pair not the best for any satellite
            continue
        
        losses=[]
        for sat in best_sats_for_g:
            tmp=sat_assign[sat]  # Ground stations in range of sat
            g_loc=np.argwhere(np.array(tmp)==g)[0][0]  # Find location of g within the list
            losses.append(sat_assign_dict[sat][g_loc][1][3])

        sat_loc=np.argwhere(np.array(losses)==min(losses))[0][0]
        best_sat=best_sats_for_g[sat_loc]

        assignment[g]=best_sat

        all_ground_pairs.remove(g)
        all_sats.remove(best_sat)
        available_sats2.remove(best_sat)


    ############################################################################
    # STEP 3
    ############################################################################
    # For the remaining available satellites, make the assignment to the first
    # best ground station that isn't already assigned
    ############################################################################

    for sat in available_sats2:
        #g_taken=True
        sat_assigned=False
        while not sat_assigned:
            g_list=sat_assign[sat]
            for g in g_list:
                if g in all_ground_pairs:
                    #g_taken=False
                    assignment[g]=sat
                    all_ground_pairs.remove(g)
                    all_sats.remove(sat)
                    sat_assigned=True
                    break
                else:
                    sat_assigned=False
            if sat not in assignment.values():   # Satellite cannot be assigned, so break the while loop
                sat_assigned=True

    ############################################################################
    # STEP 4
    ############################################################################
    # For all the remaining ground station pairs, no assignment can be made.
    ############################################################################

    for g in all_ground_pairs:
        assignment[g]=None

    
    return assignment,available_sats



def get_sat_ground_assignments(data,G_pair_labels,t):

    '''
    Loops through satellites and makes list of which pairs are in range
    and what losses are associated with each pair.
    '''
    # Last modified: 23 October 2019

    # sat_view is a dictionary that, for time t and pair i, gives a list of tuples like (sat,loss) for sats in range
    sat_view={}
    sat_view[t]={}
    for i in range(len(G_pair_labels)):
        label=str(G_pair_labels[i])
        loss_list=[]
        for z in range(len(data[t]['ground station pairs'][label]['in range'])):
            loss_list.append((data[t]['ground station pairs'][label]['in range'][z][0],data[t]['ground station pairs'][label]['in range'][z][4]))
        loss_list.sort(key=lambda loss:loss[1])
        sat_view[t][i]=[]
        sat_view[t][i]=loss_list

    # satellite_view is a dictionary that, for time t and satellite_name, gives a list of tuples (pair,loss), ordered so the lowest loss pair is first        
    satellite_view={}
    satellite_view[t]={}
    for i in range(len(G_pair_labels)):
        for z in range(len(sat_view[t][i])):
            satellite_name=sat_view[t][i][z][0]
            if satellite_name not in satellite_view[t]:
                satellite_view[t][satellite_name]=[]
                satellite_view[t][satellite_name].append((i,sat_view[t][i][z][1]))
            else:
                satellite_view[t][satellite_name].append((i,sat_view[t][i][z][1]))
    for key in satellite_view[t].keys():
        temporarylist=[]
        temporarylist=satellite_view[t][key]
        temporarylist.sort(key=lambda tup:tup[1])
        satellite_view[t][key]=temporarylist

    ######################################################################################
    # Finds lowest loss pair for each sat, puts it in a list of tuples (sat, pair, loss)
    # adds sat to list of winners, which we will use to see if it already has a pair
    # assigned to it adds pair to list of winners, which we will use to see if it
    # has already been assigned to a sat.
    ###############################################################

    winning_tups=[]
    winnersats=[]
    losersats=[]
    winnerpairs=[]
    assigned=False
    while(assigned==False):
        for key in satellite_view[t].keys():
            if key in winnersats:
                continue
            else:
                best_pairs=[]
                for p in range(len(satellite_view[t][key])):
                   if satellite_view[t][key][p][0] not in winnerpairs:
                      best_pairs.append((key,satellite_view[t][key][p][0],satellite_view[t][key][p][1]))
                      continue
                if len(best_pairs)==0:
                   if key not in losersats:
                      losersats.append(key)
                      continue
                   else:
                      continue
                else:
                   for key2 in satellite_view[t].keys():
                      if key2==key:
                         continue
                      elif key2 in winnersats:
                         continue
                      elif len(satellite_view[t][key2])==0:
                         continue
                      else:
                         for p in range(len(satellite_view[t][key2])):
                            sum=0
                            #print(p)
                            #print(best_pairs)
                            #print(satellite_view[t][key2][p])
                            if satellite_view[t][key2][p][0]==best_pairs[0][0]:
                               if satellite_view[t][key2][p][0] not in winnerpairs:
                                  best_pairs.append((key2,satellite_view[t][key2][p][0],satellite_view[t][key2][p][1]))
                                  break
                            sum+=1
                         if sum==len(satellite_view[t][key2]) and key2 not in losersats:
                            losersats.append(key2) 
                   best_pairs.sort(key=lambda tup:tup[2])
                   winning_tups.append(best_pairs[0])
                   winnersats.append(best_pairs[0][0])
                   winnerpairs.append(best_pairs[0][1])
                   if ((len(winnersats)+len(losersats))==len(satellite_view[t])):
                       assigned=True

    return sat_view, satellite_view, winning_tups



def process_assignments(assignments,data,G_pair_labels,R=1e9,h=None):

    # Last modified: 14 October 2019

    '''
    Takes the assignments provided by "filter_satellite_ground_assignments" (which are
    for a given set of ground station pairs) and provides the following
    information: 
        - Whether all ground station pairs are simultaneously in the range of a (unique) satellite
        - The times at which all ground station pairs are (and aren't) simultaneously in the range of a (unique) satellite.
        - The number of simultaneous ebits for all ground station pairs.
        - The average rates for each ground station pair (not simultaneous with other ground station pairs).

    assignments should be a dictionary with the keys being the times of the simulation.
    Then, assignments[t] should be a dictionary with the structure "pair index:satellite".

    R is the ebit source rate.
    '''

    T=len(assignments)  # Total time under consideration

    in_range={}  # Dictionary for each time step, with the value either zero or one depending on whether all ground station pairs have a unique satellite.
    times=[]  # Contains the times at which all ground station pairs have a unique satellite.
    time_gaps=[]  # Contains the times at which all ground station pairs do not have a unique satellite.
    num_ebits={}  # Dictionary for each time step, with the average number of ebits obtained *simultaneously*.
    avg_rates={} # Dictionary for each ground station pair, showing the average rate of ebits received. (does not have to be simultaneous with other ground station pairs)
    avg_losses={}


    for g in range(len(G_pair_labels)):
        avg_rates[g]=0
        avg_losses[g]=0

    avg_rate=0

    for t in range(T):
        t_assign=assignments[t]

        # Rates for individual pairs
        for g in t_assign.keys():
            if t_assign[g]==None:
                avg_rates[g]+=0/T
            else:
                if h==None:
                    sats=data[t]['ground station pairs'][str(G_pair_labels[g])]['in range']
                else:
                    sats=data[t]['ground station pairs'][str(G_pair_labels[g])]['in range'][h/1000]
                for sat in sats:
                    if sat[0]==t_assign[g]:
                        eta_tot=sat[3]
                        loss_tot=sat[4]
                        avg_rates[g]+=R*eta_tot/T
                        avg_losses[g]+=loss_tot/T
                    else:
                        continue

        # Determining whether all ground station pairs are simultaneously in range of a unique satellite, then calculating the simultaneous ebit rate
        if None in list(t_assign.values()):   # At least of the ground station pairs does not have a satellite assignment
            in_range[t]=0
            time_gaps.append(t)
            num_ebits[t]=0
            avg_rate+=0/T
        else:            # All of the ground station pairs have a satellite assignment
            in_range[t]=1
            times.append(t)
            eta_tot=1
            for g in t_assign.keys():
                if h==None:
                    sats=data[t]['ground station pairs'][str(G_pair_labels[g])]['in range']
                else:
                    sats=data[t]['ground station pairs'][str(G_pair_labels[g])]['in range'][h/1000]
                for sat in sats:
                    if sat[0]==t_assign[g]:
                        eta_tot=eta_tot*sat[3]   # All photon pairs must reach all ground station pairs in the same trial, so we multiply the transmissitivies from each satellite
                    else:
                        continue
            num_ebits[t]=R*eta_tot
            avg_rate+=num_ebits[t]/T
    
    return in_range,times,time_gaps,num_ebits,avg_rate,avg_rates,avg_losses
    



#############################################################
#### UNNECESSARY CODE
#############################################################



def get_best_satellite(data,pair,t,sats_list):

    # Last modified: 15 August 2019

    '''
    Given a pair of ground stations, this function finds the best satellite
    that is in range of that ground station pair at time t. By "best", we mean the 
    satellite for which the transmission loss is least. If there is no satellite
    in range at the given time t, then this function returns None.

    Do not specify pair as a string.

    sats_list should be a list of satellites that are known to be
    in range of the ground station pair.
    '''

    _,sats_dict=get_sats_in_range(data,[pair],t)
    sats_dict=sats_dict[0]

    L=np.array([sats_dict[sat][4] for sat in sats_list])
    
    best_loc=np.argwhere(L==np.min(L))

    best_sat=sats_list[best_loc]
    
    return best_sat

