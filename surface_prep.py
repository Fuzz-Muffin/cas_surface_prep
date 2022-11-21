import sys, argparse, pdb, ast, pickle, os, random, datetime, subprocess
import numpy as np
import MDAnalysis as md
import argparse as ap
import polypy as pp

#global things
rng = np.random.default_rng(666)
surface_depth = 2.0

def main(args):
    fbase = args.infile.split('/')[-1].split('.')[0]
    open_rings = args.open_rings
    water = args.water
    fixed_oh = args.fixed_oh
    harmonic_oh = args.harmonic_oh

    # get input xyz file
    u = md.Universe(args.infile)
    natoms = u.atoms.n_atoms
    coords = u.atoms.positions
    names = u.atoms.names

    # get box dimensions because MDAnalysis is too dumb to find them
    with open(args.infile,'r') as fi:
        for i, line in enumerate(fi.readlines()):
            if i == 1:
                if 'Lattice=' in line:
                    # do later
                    sys.exit('NOT WORKING FIX LATER')
                else:
                    box = np.array([float(xx) for xx in line.strip().split()])

    # set atoms so that lowest atom is at z=0
    coords[:,-1] -= coords[:,-1].min()
    # give box 60Å of vacuum gap
    box[-1] = coords[:,-1].max() + 60
    # wrap everything so x,y positions are between 0-box
    coords[:,:-1] -= box[:-1] * 0.5
    coords[:,:-1] -= box[:-1] * np.round(coords[:,:-1]/box[:-1])
    coords[:,:-1] += box[:-1] * 0.5

    outfile = fbase + '_pre'
    write_xyz_file(coords, names, box, outfile)

    # find rings
    ring_dict = calc_rings(outfile+'.xyz')

    # create or read in the topology
    topofile = fbase + '_topo.pickle'
    if os.path.isfile(topofile):
        print('Reading initial topology...')
        with open(topofile, 'rb') as fi:
            nn_list = pickle.load(fi)

    else:
        print('Calculating initial topology...')
        nn_list = calc_neighbours(coords, names, box)
        with open(topofile, 'wb') as fi:
            pickle.dump(nn_list, fi)

    ## find singly coordinated oxygen close (5Ang) within surface
    print('Locating surfaces...')
    top_surf, bot_surf = find_surfaces(coords, box)

    print('Determining surface modifications...')
    add_oh_to_x, remove_o, protonate, remove_ca = find_oh_sites(coords, top_surf, bot_surf, names, nn_list, ring_dict, box, open_rings)

    # XYZ that shows which oxygens need to be removed
    outfile = fbase + '_flagged_O'
    tmp_names = names.copy()
    if open_rings:
        tmp_names[np.array(remove_o)] = 'X'
        tmp_names[np.array(add_oh_to_x)] = 'Y'

    tmp_names[np.array(remove_ca)] = 'CAX'
    tmp_names[np.array(protonate)] = 'Z'
    write_xyz_file(coords, tmp_names, box, outfile)

    # attach OH groups to mark sites
    print('Terminating surfaces...')
    new_coords, new_names, new_nn_list, hydroxyl_bonds = hydroxylate_surface(coords, names, box, add_oh_to_x, protonate, remove_ca, remove_o, nn_list, fbase)


    # remove the deleted oxygens and Ca
    remove_mask = ~np.logical_or(np.isin(np.arange(new_coords.shape[0], dtype=int), remove_o),
                                 np.isin(np.arange(new_coords.shape[0], dtype=int), remove_ca) )

    # mapping between new id and old id
    id_map = np.arange(new_coords.shape[0], dtype=int)
    id_map = id_map[remove_mask]
    def map_id(old_id):
        return np.where(id_map == old_id)[0][0]

    # use harmonic bonds between O1-H1
    if harmonic_oh:
        bonds = []
        blah = np.array([new_names[j] for j in hydroxyl_bonds])
        for ids in hydroxyl_bonds:
            nn = new_nn_list[ids[0]]
            nn_t = new_names[nn]
            h_id = nn[list(nn_t).index('H1',1)]
            bonds.append([2, map_id(ids[0])+1, map_id(h_id)+1])

    # if fixed_oh is True then put in harmonic bonds between O1-X
    if fixed_oh:
        if not harmonic_oh:
            bonds = []

        blah = np.array([new_names[j] for j in hydroxyl_bonds])
        for ids, types in zip(hydroxyl_bonds, blah):
            if types[-1] == 'Al':
                bt = 3
            elif types[-1] == 'Si':
                bt = 4
            bonds.append([bt, map_id(ids[0])+1, map_id(ids[1])+1])


    new_coords = new_coords[remove_mask]
    new_names = new_names[remove_mask]
    new_natoms = new_coords.shape[0]

    print('Writing slab output to file...')
    outfile = fbase + '_surfaceterminated'
    write_pdb_file(new_coords, new_names, box, outfile)
    write_xyz_file(new_coords, new_names, box, outfile)
    if fixed_oh:
        write_lammps_file(new_coords, new_names, box, outfile, bonds= np.array(bonds))
    else:
        write_lammps_file(new_coords, new_names, box, outfile)

    if water:
        nwater = int(np.round( 0.02 * (box[0]-2.) * (box[1]-2.) * (box[2] - coords[:,-1].max() - 2.) ))
        print(f'Pouring {nwater} waters...')
        # find packmol executable
        packmol = subprocess.run(['which','packmol'], capture_output=True).stdout.strip()

        waterfile = 'SPCE.pdb'
        #if not os.path.isfile(waterfile):
        onewater = 'REMARK SPC/E water\n'
        onewater += 'ATOM    229 OW   M1   77         5.529  12.786   1.902  1.00  0.00           O\n'
        onewater += 'ATOM    230 HW   M1   77         4.624  12.731   1.481  1.00  0.00           H\n'
        onewater += 'ATOM    231 HW   M1   77         5.682  13.713   2.244  1.00  0.00           H\n'
        onewater += 'END'

        with open(waterfile, 'w') as fo:
            fo.write(onewater)

        #write packmol input
        packinp = 'waterpack.inp'
        #if not os.path.isfile(packinp):
        with open(packinp, 'w') as fo:
            fo.write('tolerance 2.0\n')
            fo.write('output bulk_water.pdb\n')
            fo.write(f'structure {waterfile}\n')
            fo.write(f'  number {nwater}\n')
            fo.write(f'  inside box 0. 0. 35.5 46.5 46.5 91.0\n')
            fo.write('end structure')

        with open(packinp, 'r') as fi:
            subprocess.run(['packmol'], stdin=fi, shell=True, capture_output=True)

        # open water coordinates and add them to the slab
        u2 = md.Universe('bulk_water.pdb')
        w_coords = u2.atoms.positions
        w_names = u2.atoms.names

        # clean up a little
        if os.path.isfile(packinp):
            os.remove(packinp)

        if os.path.isfile(waterfile):
            os.remove(waterfile)

        if os.path.isfile('bulk_water.pdb'):
            os.remove('bulk_water.pdb')

        old_natoms = new_names.shape[0]
        w_ids = np.arange(1,w_names.shape[0]+1) + old_natoms

        # calculate the water topology, we make some correct assumptions to make this easier
        if not fixed_oh:
            bonds = []

        for i, ind in enumerate(w_ids):
            if i%3 == 0:
                bonds.append([1, ind, ind+1])
                bonds.append([1, ind, ind+2])

        angles = [[1, ind+1, ind, ind+2] for i,ind in enumerate(w_ids) if i%3 == 0]
        bonds = np.array(bonds)
        angles = np.array(angles)

        print('Writing water interfaced slab output to file...')
        outfile = fbase + '_surfaceterminated_wet'
        new_coords = np.row_stack((new_coords, w_coords))
        new_names = np.concatenate((new_names, w_names))

        write_pdb_file(new_coords, new_names, box, outfile)
        write_xyz_file(new_coords, new_names, box, outfile)

        write_lammps_file(new_coords, new_names, box, outfile, bonds= bonds, angles= angles)

# Use polypy to calculate the SP rings
def calc_rings(inxyzfile):
    rings = {}
    base = inxyzfile.split('/')[-1].split('.')[0]
    pp.main([inxyzfile, '-p', '-g', '-d 6', '-b'])

    with open(f'{base}.nfo','r') as fi:
        for line in fi:
            if '[ring elements]' in line:
                ring_string = next(fi, '').strip()
                break

    ring4 = []
    ring6 = []
    for r in list(ast.literal_eval(ring_string)):
        j = []
        for atom in r:
            try:
                j.append(atom)
            except ValueError:
                pass

        ringnumber = len(j)
        if ringnumber == 4:
            ring4.append(j)
        elif ringnumber == 6:
            ring6.append(j)

    rings[4] = ring4
    rings[6] = ring6

    return rings

def distances(x0, x, box, pbc=np.array([True, True, True])):
    # xo is a position of one atom, x1 is an array of positions
    # use the pbc bool mask to set the periodicity
    delta = np.abs(x0 - x)
    delta[:,pbc] -= box[pbc] * np.round(delta[:,pbc]/box[pbc])
    return(np.sqrt((delta ** 2).sum(axis=-1)))

def calc_neighbours(coords, names, box):
    from itertools import combinations_with_replacement as comb
    # define cutoffs between unique element pairs
    name_cutoff = {}
    elements = ['Al','Ca','H','O','Si','OW','HW']
    elements = sorted(elements)
    pairs = list(comb(elements, 2))
    for (i_el, j_el) in pairs:
        term = i_el + '-' + j_el
        name_cutoff[term] = 0.0
        if (i_el == 'Si') or (i_el == 'Al'):
            if (j_el == 'Si') or (j_el == 'Al'):
                name_cutoff[term] = 1.7
            elif j_el == 'Ca':
                name_cutoff[term] = 1.5
            elif j_el == 'O':
                name_cutoff[term] = 2.4
            elif j_el == 'H':
                name_cutoff[term] = 1.3

        elif i_el == 'O':
            if (j_el == 'Si') or (j_el == 'Al'):
                name_cutoff[term] = 2.4
            elif j_el == 'Ca':
                name_cutoff[term] = 3.0
            elif j_el == 'O':
                name_cutoff[term] = 1.81
            elif j_el == 'H':
                name_cutoff[term] = 1.3

        elif i_el == 'Ca':
            if (j_el == 'Si') or (j_el == 'Al'):
                name_cutoff[term] = 1.5
            elif j_el == 'Ca':
                name_cutoff[term] = 1.5
            elif j_el == 'O':
                name_cutoff[term] = 3.0
            elif j_el == 'H':
                name_cutoff[term] = 1.3

        elif i_el == 'H':
            if (j_el == 'Si') or (j_el == 'Al'):
                name_cutoff[term] = 1.81
            elif j_el == 'Ca':
                name_cutoff[term] = 2.5
            elif j_el == 'O':
                name_cutoff[term] = 1.81
            elif j_el == 'H':
                name_cutoff[term] = 1.3

        elif i_el == 'HW':
            if j_el == 'OW':
                name_cutoff[term] = 1.25

        elif i_el == 'OW':
            if j_el == 'HW':
                name_cutoff[term] = 1.25

    neigh_list = []
    for i, (i_coord, i_el) in enumerate(zip(coords,names)):
        dists = distances(i_coord, coords, box)
        cutoffs = np.array([name_cutoff[sorted([i_el,j_el])[0] + '-' + sorted([i_el,j_el])[1]] for j_el in names])
        bonded = (dists < cutoffs) & (dists > 1e-10)
        bonded_types = names[bonded]
        if len(bonded_types)>0:
            bonded_inds = np.where(bonded)[0]
        else:
            bonded_inds = []
        neigh_list.append(list(bonded_inds))

    return neigh_list

# find surfaces from density profile in z
def find_surfaces(coords, box):
    ## average position
    avgz = np.average(coords[2])
    ## create bins (5x5AA)
    binsize = 5
    xnumbins,ynumbins = int(box[0]/binsize), int(box[1]/binsize)
    bins = {}
    for x in range(0,xnumbins+1):
        for y in range(0,ynumbins+1):
            bins[x,y] = []
    # bin coords
    for atom in coords:
        xbin = int(atom[0]/(binsize))
        ybin = int(atom[1]/(binsize))
        bins[xbin,ybin].append([atom[0],atom[1],atom[2]])
    ## maximum + and - distance from avg position for each bin
    top_surface_coords = []
    bottom_surface_coords = []
    for x,y in bins:
        toplimit = np.array(bins[x,y])[:,-1].max()
        botlimit = np.array(bins[x,y])[:,-1].min()
        top_surface_coords.append([x*binsize, y*binsize, toplimit])
        bottom_surface_coords.append([x*binsize, y*binsize, botlimit])

    return top_surface_coords, bottom_surface_coords

# install OH groups
def find_oh_sites(coords, top_surf, bot_surf, names, nn_list, rings, box, open_rings=False):
    print(f'    Surface depth is {surface_depth} Å')
    # define surface grid
    xy_grid = np.array(top_surf)[:,:-1] + 5
    top_surf = np.array(top_surf)[:,-1]
    bot_surf = np.array(bot_surf)[:,-1]

    protonate = []
    add_oh_to_x = []

    # go through each ring
    ring_mem_list = rings[4] + rings[6]
    ring_size_list = [4 for _ in range(len(rings[4]))] + [6 for _ in range(len(rings[6]))]

    # non-Ca nearest neighbour list for each atom
    network_nn =[[blah for blah in n if names[blah] != 'Ca'] for n in nn_list]
    # non-Ca coordination number
    z = np.array([len(n) for n in network_nn])
    zz = np.array([len(n) for n in nn_list])
    print('Oxygen coordination stats:')
    print(np.unique(z[names == 'O'], return_counts=True))

    # first protonate all NBOs within 5A of the surfaces
    nbo = np.logical_and(z == 1, names == 'O')
    free_o = np.logical_and(z == 0, names == 'O')

    # if there are x free O, then remove x more Ca so that we can just delete them
    if free_o.sum()>0:
        remove_o = [i for i,test in enumerate(free_o) if test]
        print(f'    Removing {free_o.sum()} free oxygens')
    else:
        remove_o = []

    surf_nbo = nbo.copy()
    under_coord_si = np.logical_and(z < 4, names == 'Si')
    under_coord_al = np.logical_and(z < 4, names == 'Al')
    under_coord_x = np.logical_or(under_coord_si, under_coord_al)
    # check first if Al and Si that are under coord are near surface
    for i, isunder in enumerate(nbo):
        grid_inds = []
        if isunder:
            pos = coords[i]
            delta = xy_grid - pos[:-1]
            grid_inds.append(np.all(np.logical_and( delta>0, delta<=5 ), axis=-1))
            top_surf_delta = np.abs(top_surf[grid_inds[-1]] - pos[-1])
            bot_surf_delta = np.abs(pos[-1] - bot_surf[grid_inds[-1]])

            istop = False; isbot = False
            if top_surf_delta < surface_depth: istop = True
            if bot_surf_delta < surface_depth: isbot = True
            if (not istop) and (not isbot): surf_nbo[i] = False

    if np.mod(surf_nbo.sum(), 2) > 0:
        surf_nbo_ind = np.where(surf_nbo == True)[0][:-1]
    else:
        surf_nbo_ind = np.where(surf_nbo == True)[0]

    protonate.extend(surf_nbo_ind)
    num_ca_remove = int(surf_nbo_ind.size/2)
    # now account for the number of removed free O
    num_ca_remove += len(remove_o)

    # target only bulk Ca, too much surface rearrangement otherwise
    print('Ca coordination stats:')
    print(np.unique(zz[names == 'Ca'], return_counts=True))
    # print out coordination stats
    with open('coord_stats.log', 'w') as fo:
        fo.write('Element 0 1 2 3 4 5 6')
        for el in np.unique(names):
            fo.write(f'\n{el} ' + ' '.join([str(int((z[names==el]==num).sum())) for num in range(7)]))

    ca_mask = names == 'Ca'
    num_ca = ca_mask.sum()
    ca_idx = np.where(ca_mask)[0]
    potential_remove_ca = ca_mask

    #================================#
    # THIS CODE ONLY SELECTS BULK Ca #
    #================================#
    #for ind in ca_idx:
    #    pos = coords[ind]
    #    delta = xy_grid - pos[:-1]
    #    grid_inds.append(np.all(np.logical_and( delta>0, delta<=5 ), axis=-1))
    #    top_surf_delta = np.abs(top_surf[grid_inds[-1]] - pos[-1])
    #    bot_surf_delta = np.abs(pos[-1] - bot_surf[grid_inds[-1]])

    #    istop = False; isbot = False
    #    if top_surf_delta < surface_depth: istop = True
    #    if bot_surf_delta < surface_depth: isbot = True
    #    if (istop) or (isbot): potential_remove_ca[ind] = False

    to_remove = rng.choice(np.arange(coords.shape[0])[potential_remove_ca], num_ca_remove, replace=False)
    ca_remove = list(np.arange(coords.shape[0])[to_remove])
    print(f'    NBOs in surface: {len(surf_nbo_ind)}')
    print(f'    Removing {int(len(surf_nbo_ind)/2)} + {free_o.sum()} = {len(ca_remove)} Ca in total')

    if open_rings:
        two_rings = 0
        three_rings = 0
        to_save = np.full(coords.shape[0], 0)

        deleted_o = []
        for ii, mems in enumerate(ring_mem_list):
            # we want to only break a link ooxygen between Si,Al with a coord of 2
            size = len(mems)
            mem_pos = coords[mems]
            mem_name = names[mems]
            network_z = z[mems]
            o_mask_network = np.logical_and(mem_name == 'O', network_z == 2)
            o_mask = mem_name == 'O'

            # check if any part of ring is near the local surface
            top_surf_delta = np.zeros(len(mems))
            bot_surf_delta = np.zeros(len(mems))
            grid_inds = []
            for i, atom in enumerate(mem_pos):
                delta = xy_grid - atom[:-1]
                grid_inds.append(np.all(np.logical_and( delta>0, delta<=5 ), axis=-1))
                top_surf_delta[i] = np.abs(top_surf[grid_inds[-1]] - atom[-1])
                bot_surf_delta[i] = np.abs(atom[-1] - bot_surf[grid_inds[-1]])

            # any atoms in the ring within <surface_depth> Å of the surface?
            if np.any(top_surf_delta < surface_depth):
                istop = True
            else:
                istop = False

            if np.any(bot_surf_delta < surface_depth):
                isbot = True
            else:
                isbot = False

            # check if any of the oxygens have been already deleted
            # if yes, then we don't want to touch this ring
            already_deleted = [True if el in deleted_o else False for el in mems]
            if np.any(already_deleted):
                istop = False
                isbot = False

            if (istop) or (isbot):
                mems_arr = np.array(mems)


                # second, check if there is a two-fold coord O
                if o_mask_network.sum() == 1:
                    o_id = int(mems_arr[o_mask_network])
                    remove_o.append(o_id)
                    deleted_o.append(o_id)
                    add_oh_to_x.extend(network_nn[o_id])
                    if size == 4:
                        two_rings += 1
                        to_save[mems] = 2
                    elif size == 6:
                        three_rings += 1
                        to_save[mems] = 3

                elif o_mask_network.sum() > 1:
                    real_neighs = [len(nn_list[ooo]) for ooo in mems_arr[o_mask_network]]
                    if 2 in real_neighs:
                        ind = real_neighs.index(2)
                        o_id = int(mems_arr[o_mask_network][ind])
                        remove_o.append(o_id)
                        deleted_o.append(o_id)
                        add_oh_to_x.extend(network_nn[o_id])
                        if size == 4:
                            two_rings += 1
                            to_save[mems] = 2
                        elif size == 6:
                            three_rings +=1
                            to_save[mems] = 3
                    else:
                        o_pos_z = mem_pos[o_mask_network][:,-1]
                        if istop:
                            o_surf_ind = o_pos_z.argmax()
                        elif isbot:
                            o_surf_ind = o_pos_z.argmin()

                        o_id = int(mems_arr[o_mask_network][o_surf_ind])
                        remove_o.append(o_id)
                        deleted_o.append(o_id)
                        add_oh_to_x.extend(network_nn[o_id])
                        if size == 4:
                            two_rings += 1
                            to_save[mems] = 2
                        elif size == 6:
                            three_rings +=1
                            to_save[mems] = 3

                #else:
                #    o_pos_z = mem_pos[o_mask][:,-1]
                #    if istop:
                #        o_surf_ind = o_pos_z.argmax()
                #    elif isbot:
                #        o_surf_ind = o_pos_z.argmin()

                #    o_id = int(mems_arr[o_mask][o_surf_ind])
                #    remove_o.append(o_id)
                #    add_oh_to_x.extend(network_nn[o_id])
                #    if len(network_nn[o_id])> 2:
                #        pdb.set_trace()
                #    if size == 4:
                #        two_rings += 1
                #        to_save[mems] = 2
                #    elif size == 6:
                #        three_rings +=1
                #        to_save[mems] = 3

        print(f'    Opened {two_rings} 2-mem rings and {three_rings} 3-mem rings, so {int(len(add_oh_to_x)/2)} in total')
        is_ring = np.nonzero(to_save)
        write_xyz_file(coords[is_ring], to_save[is_ring], box, 'surface_rings')

    return(add_oh_to_x, remove_o, protonate, ca_remove)

######################################################################
# https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sp
######################################################################
def fibonacci_sphere(samples=70):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        vec = np.array([x,y,z])
        vec = vec / np.linalg.norm(vec)
        points.append(vec)

    return(np.array(points))

######################################################################
## Given a list of Al/Si to add OH to, add OH
## - first try calculates an oxygen position via averaging the NN
##      of the neighboring oxygens and mirroring across the Al/Si
## - check if the placed oxygen overlaps
######################################################################
def hydroxylate_surface(old_coords, old_names, box, add_oh_to_x, protonate, remove_ca, remove_o, old_nn_list, fbase):
    add_coords = []
    add_names = []
    add_nn = []
    mid = old_coords[:,-1].mean()
    natoms = old_coords.shape[0]
    already_oh = []
    hydroxyl_bonds = []

    # when fitting in new groups, make sure to remove the flagged ones
    not_removed_mask = np.isin(np.arange(old_names.shape[0]), np.array(remove_ca), invert=True)
    not_removed_mask = np.logical_and(not_removed_mask, np.isin(np.arange(old_names.shape[0]), np.array(remove_o), invert=True))

    # protonate surface NBO sites
    print('    Protonating NBOs')
    for site, pos in zip(protonate, old_coords[protonate]):
        zpos = pos[-1]
        if add_coords:
            all_coords = np.row_stack((old_coords[not_removed_mask],np.array(add_coords)))

        else:
            all_coords = old_coords[not_removed_mask]

        # now remove the origin atom
        all_coords = all_coords[~(all_coords == pos).all(1)]
        if zpos > mid:
            flip = 1.0

        else:
            flip = -1.0

        p1 = pos + np.array([0.0,0.0,flip*1.])
        ps = np.array([p1])

        dists = distances(pos, all_coords, box)
        mask = np.logical_and( dists<5, dists>1e-5 )
        near_pos = all_coords[mask]
        good_vecs, stat = check_and_move(pos, ps, near_pos, box)
        if stat:
            new_h = good_vecs[0]

        else:
            sys.exit('BAIL OUT')

        # find the root to the OH
        nn = old_nn_list[site]
        nn_names = old_names[nn]
        mask = ~(nn_names == 'Ca')
        if mask.sum() == 1:
            hydroxyl_bonds.append([site, np.array(nn)[mask][0]] )
        else:
            pdb.set_trace()

        add_coords.append(new_h)
        add_names.extend(['H1'])
        add_nn.append([site])
        old_nn_list[site].append(natoms)
        already_oh.append(site)

        natoms += 1

    # add OH groups to (Si, Al) sites
    print('    Hydroxylating opened ring sites')

    hydrox_f = fbase + '_hydrodxyl.pickle'
    if os.path.isfile(hydrox_f):
        print('        Read in previously calculated hydroxyl positions')
        with open(hydrox_f, 'rb') as fi:
            hydroxyl_loc = pickle.load(fi)
            for k, (site, pos, loc) in enumerate(zip(add_oh_to_x, old_coords[add_oh_to_x], hydroxyl_loc)):
                add_coords.append(loc[0])
                add_coords.append(loc[1])
                add_names.extend(['O1','H1'])
                add_nn.append([site,natoms+1])
                add_nn.append([natoms])
                already_oh.append(site)
                hydroxyl_bonds.append([natoms, site])
                natoms += 2

    else:
        print('        Calculating hydroxyl positions')
        sphere_points = fibonacci_sphere(samples=5000)
        write_xyz_file(sphere_points, np.full(sphere_points.shape[0], 'X'), box, 'points_sphere.xyz')
        hydroxyl_loc = []
        for k, (site, pos) in enumerate(zip(add_oh_to_x, old_coords[add_oh_to_x])):
            print(f'        {k}th hydroxyl')
            zpos = pos[-1]
            if zpos > mid:
                flip = 1.0
            else:
                flip = -1.0

            p1 = pos + np.array([0.0,0.0,flip*1.3])
            p2 = pos + np.array([0.0,0.0,flip*1.3]) + np.array([1/np.sqrt(3.), 1/np.sqrt(3.), flip/np.sqrt(3.)])

            ps = np.array([p1,p2])

            if add_coords:
                all_coords = np.row_stack((old_coords[not_removed_mask],np.array(add_coords)))
            else:
                all_coords = old_coords[not_removed_mask]

            # now remove the origin atom
            all_coords = all_coords[~(all_coords == pos).all(1)]

            dists = distances(pos, all_coords, box)
            mask = np.logical_and( dists<6, dists>1e-5 )
            mask[site] = False
            near_pos = all_coords[mask]
            is_done = False
            tol = 2.45
            if mask.sum()>0:
                while not is_done:
                    stat = False
                    for p1 in sphere_points:
                        p2 = p1*2.3
                        p1 = p1*1.4

                        p1 += pos
                        p2 += pos
                        dists = []
                        for p in [p1,p2]:
                            dists.extend(distances(p, near_pos, box))

                        dists = np.array(dists)
                        mask = np.logical_and( dists<tol, dists>1e-5 )
                        if np.all(mask == False):
                            good_vecs = [p1,p2]
                            stat = True
                            is_done = True
                            break

                    if stat:
                        new_o = good_vecs[0]
                        new_h = good_vecs[1]
                        is_done == True
                        print(f'            tol = {tol:.2f}')

                    else:
                        tol -= 0.1

            else:
                new_o = good_vecs[0]
                new_h = good_vecs[1]

            ## ALTERNATE METHOD NOT CURRENTLY USED
            ## randomly move the group around
            #good_vecs, stat = check_and_move(pos, ps, near_pos, box)

            hydroxyl_loc.append([new_o,new_h])
            add_coords.append(new_o)
            add_coords.append(new_h)
            add_names.extend(['O1','H1'])
            add_nn.append([site,natoms+1])
            add_nn.append([natoms])
            already_oh.append(site)
            hydroxyl_bonds.append([natoms, site])
            natoms += 2

        with open(hydrox_f, 'wb') as fi:
            pickle.dump(hydroxyl_loc, fi)

    coords = np.array(list(old_coords) + add_coords)
    names = old_names
    # change oxygen type of protonated NBO sites
    names[protonate] = 'O1'
    names = np.array(list(names) + add_names)
    nn_list = old_nn_list + add_nn
    return coords, names, nn_list, np.array(hydroxyl_bonds)

def rotate_about_vec(u,a,theta):
    import quaternion
    # rotate vector u, by an angle theta, about an arbitrary vector a, using quaternions. Nifty aye?
    a = a / np.linalg.norm(a)
    q1 = np.quaternion(0.,u[0],u[1],u[2])
    q2 = np.quaternion(np.cos(theta/2.), a[0] * np.sin(theta/2.), a[1] * np.sin(theta/2.), a[2]*np.sin(theta/2.))
    q3 = q2 * q1 * np.conjugate(q2)
    return quaternion.as_float_array(q3)[1:]

def check_and_move(o, ps, near_pos, box, pbc=[True, True, True], sep=1.80):
    dists = []
    for p in ps:
        dists.extend(distances(p, near_pos, box))
    count = 0
    status = True
    while np.any(np.array(dists) < sep):
        dists = []
        count += 1
        if count > 1000000:
            status = False
            p = np.empty(3)*np.nan
            break
        # if any atom is too close, move the whole lot about some random vector v
        v = np.array([random.random() for _ in range(3)])
        if ps.shape[0]==3:
            ps = rotate_about_vec(ps-o, v, np.pi/2.)
            ps += o
            dists = distances(ps, near_pos, box, pbc)
        else:
            for i,p in enumerate(ps):
                if i==0: dists = []
                p = rotate_about_vec(p-o, v, np.pi/2.)
                p += o
                ps[i] = p
                dists.extend(distances(p, near_pos, box, pbc))

        #print(np.array(dists)[np.array(dists) < sep])
    return ps, status

def write_pdb_file(Coords,AtomTypes,ABC,fbase):
    newlines = []
    fname = fbase+'.pdb'
    for i in range(0,len(Coords)):
        newlines.append("{:6s}{:5d} {:^4s} {:3s}  {:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}".format("ATOM",i ,str(AtomTypes[i]),"UNL",1,Coords[i][0],Coords[i][1],Coords[i][2],1.00,0.0,str(AtomTypes[i])))

    with open(fname, 'w') as fo:
        fo.write("AUTHOR    FV SURFACE PREP\n")
        fo.write("CRYST1 {:8.3f} {:8.3f} {:8.3f} {:6.2f} {:6.2f} {:6.2f} P1          1\n".format(ABC[0],ABC[1],ABC[2],90.0,90.0,90.0))
        for line in newlines:
            fo.write("%s\n"%(line))

    return

def read_pdb_file(filename):
    coords = []
    names = []
    with open(filename, 'r') as fi:
        for line in fi.readlines():
            if 'ATOM' in line:
                tmp = line.strip().split()
                names.append(tmp[2])
                coords.append([float(ii) for ii in tmp[6:9]])

    coords = np.array(coords)
    names = np.array(names)
    return coords, names


def write_xyz_file(coords, names, box, fbase):
    natoms = coords.shape[0]
    fname = fbase+'.xyz'
    with open(fname, 'w') as fo:
        fo.write(f'{natoms}\n')
        fo.write(' '.join(map(str,box)))
        for t,p in zip(names,coords):
            fo.write(f'\n{t:2} {p[0]:6.4} {p[1]:6.4} {p[2]:6.4}')

    return

def write_lammps_file(coords, names, box, fbase, bonds= None, angles= None, improps= None, diheds= None):
    # get the date and time
    now = datetime.datetime.now()
    outfile = fbase + '.data'

    natoms = coords.shape[0]
    elements = ['Ca','Si','Al','O','Fe','Mg','OW','HW','O1','H1']
    masses = [40.078, 28.085, 26.981, 15.998, 55.845, 24.305, 15.998, 1.008, 15.998, 1.008]

    if bonds is None:
        nbonds = 0
        nbondtypes = 0
    else:
        nbonds = bonds.shape[0]
        nbondtypes = np.unique(bonds[:,0]).size

    if angles is None:
        nangles = 0
        nangletypes = 0
    else:
        nangles = angles.shape[0]
        nangletypes =  np.unique(angles[:,0]).size

    if diheds is None:
        ndiheds = 0
        ndihedtypes = 0

    if improps is None:
        nimprops = 0
        nimproptypes = 0

    atom_type = np.zeros(natoms, dtype=int)
    atom_mol = np.full(natoms, 1, dtype=int)
    atom_q = np.full(natoms, 0.0, dtype=float)
    atom_image = np.full((natoms,3), 0, dtype=int)

    for i, el in enumerate(elements):
        atom_type[names == el] = i+1

    xlo, ylo, zlo = [0.0 for _ in range(3)]
    xhi, yhi, zhi = box

    with open(outfile, 'w') as fo:
        header = f'#lammps data file written by fv script {now.strftime("%H:%M:%S %d %b %y")}'
        fo.write(header + '\n\n')
        fo.write(str(natoms) + ' atoms\n')
        fo.write(str(nbonds) + ' bonds\n')
        fo.write(str(nangles) + ' angles\n')
        fo.write(str(ndiheds) + ' dihedrals\n')
        fo.write(str(nimprops) + ' impropers\n\n')

        fo.write('10 atom types\n')
        fo.write(f'4 bond types\n')
        fo.write(f'1 angle types\n')
        fo.write(str(ndihedtypes) + ' dihedral types\n')
        fo.write(str(nimproptypes) + ' improper types\n\n')

        fo.write(str(xlo) + ' ' + str(xhi) + ' xlo xhi\n')
        fo.write(str(ylo) + ' ' + str(yhi) + ' ylo yhi\n')
        fo.write(str(zlo) + ' ' + str(zhi) + ' zlo zhi\n\n')

        mass_string = 'Masses\n'
        for i, (el, mass) in enumerate(zip(elements, masses)):
            mass_string += f'\n{i+1}  {mass}  #  {el}'

        mass_string + '\n\n'
        fo.write(mass_string)

        outstring ='\n\nAtoms # full\n\n'
        for i in range(natoms):
            outstring += f'{i+1} {atom_mol[i]} {atom_type[i]} {atom_q[i]} ' + ' '.join(map(str,coords[i,:])) + ' ' + ' '.join(map(str,atom_image[i,:])) + ' \n'
        fo.write(outstring)

        if bonds is not None:
            outstring = '\nBonds\n\n'
            for i,bond in enumerate(bonds):
                outstring += str(i+1) + ' ' + ' '.join(map(str,bonds[i,:]))+ ' \n'
            fo.write(outstring)

        if angles is not None:
            outstring = '\nAngles\n\n'
            for i,angle in enumerate(angles):
                outstring += str(i+1) + ' ' + ' '.join(map(str,angles[i,:]))+ ' \n'
            fo.write(outstring)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Prepare CAS surface for water interface')
    parser.add_argument('--infile', help='<Required> input file name (xyz)', required=True)
    parser.add_argument('--water', action='store_true', help='optional flag to specify if water is included. If flagged water coordinates are read from shifted-ow.pdb. Default is False')
    parser.add_argument('--open_rings', action='store_true', help='Open two-, three-member O linkages at the surface, default is False')
    parser.add_argument('--fixed_oh', action='store_true', help='Setup fixed bonds between OH groups linking sites')
    parser.add_argument('--harmonic_oh', action='store_true', help='Use harmonic bonds between O1-H1')
    args = parser.parse_args(sys.argv[1:])
    # run main
    main(args)
