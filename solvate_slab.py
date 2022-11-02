import sys, argparse, pdb, ast, pickle, os, random, datetime, subprocess
import numpy as np
import MDAnalysis as md
import argparse as ap
import polypy as pp
rng = np.random.default_rng(666)

def main(args):
    fbase = args.infile.split('/')[-1].split('.')[0]

    # get input xyz file
    u = md.Universe(args.infile)
    natoms = u.atoms.n_atoms
    coords = u.atoms.positions
    atypes = u.atoms.types
    box = u.dimensions[:3]

    # no names in lammps data format
    # names = u.atoms.names

    # if there are slab atoms crossing z boundary then move them up
    if coords[:,-1].max() > box[-1]*0.5:
        val = coords[:,-1].max()
        trans = np.abs(box[-1] - val) + 1.0
        new_coords = coords
        new_coords[:,-1] += trans
        u.atoms.positions = new_coords
        u.atoms.wrap()
        coords = u.atoms.positions

    u.dimensions[2] += 0.0
    box = u.dimensions[:3]

    slab_top = coords[:,-1].max()

    ## find loose O atoms
    #o_mask = atypes == 4
    #network_nn =[[blah for blah in n if atypes[blah] != '1'] for n in nn_list]
    ## network coordination number
    #z = np.array([len(n) for n in network_nn])
    ## total coordination number
    #zz = np.array([len(n) for n in nn_list])
    ## list of free O atom ids
    #mask = np.logical_and(o_mask, z == 0)
    #free_o = np.argwhere(mask == True) + 1

    nwater = int(np.round( 0.033 * (box[0]-2.) * (box[1]-2.) * (box[2] - slab_top - 2.) ))
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
        fo.write(f'  inside box 0. 0. {slab_top + 2} {box[0]-2.} {box[1]-2.} {box[2]-2.}\n')
        fo.write('end structure')

    with open(packinp, 'r') as fi:
        subprocess.run(['packmol'], stdin=fi, shell=True)

    # open water coordinates and add them to the slab
    u2 = md.Universe('bulk_water.pdb')
    w_coords = u2.atoms.positions
    w_names = u2.atoms.names
    u2.atoms.types = ['7','8','8'] * int(u2.atoms.n_atoms/3)

    # clean up a little
    if os.path.isfile(packinp):
        os.remove(packinp)

    if os.path.isfile(waterfile):
        os.remove(waterfile)

    if os.path.isfile('bulk_water.pdb'):
        os.remove('bulk_water.pdb')

    old_natoms = u.atoms.n_atoms
    w_ids = np.arange(1,u2.atoms.n_atoms) + old_natoms

    # calculate the water topology, we make some correct assumptions to make this easier
    bonds = []
    #for i, ind in enumerate(w_ids):
    #    if i%3 == 0:
    #        bonds.extend([(ind, ind+1),(ind, ind+2)])

    for i in range(0,u2.atoms.n_atoms, 3):
        bonds.extend([(i, i+1), (i, i+2)])


    angles = [(i+1, i, i+2) for i in range(u2.atoms.n_atoms) if i%3 == 0]
    u2.add_bonds(bonds, types = ['1' for _ in range(len(bonds))])
    u2.add_angles(angles, types = ['1' for _ in range(len(angles))])

    print('Writing water interfaced slab output to file...')
    outfile = fbase + '_surfaceterminated_cleanup_wet.data'
    new_u = md.Merge(u.atoms, u2.atoms)
    new_u.add_dihedrals([])
    new_u.add_impropers([])
    new_u.dimensions = u.dimensions
    new_u.add_TopologyAttr('charges', [0.0 for _ in range(new_u.atoms.n_atoms)])

    md.Writer(outfile).write(new_u)

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
    elements = ['Al','Ca','H','O','Si','OW','HW','O1','H1']
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
            elif j_el == 'O1':
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
            elif j_el == 'O1':
                name_cutoff[term] = 1.81
            elif j_el == 'H':
                name_cutoff[term] = 1.3

        elif i_el == 'O1':
            if (j_el == 'Si') or (j_el == 'Al'):
                name_cutoff[term] = 2.4
            elif j_el == 'Ca':
                name_cutoff[term] = 3.0
            elif j_el == 'O':
                name_cutoff[term] = 1.81
            elif j_el == 'O1':
                name_cutoff[term] = 1.81
            elif j_el == 'H':
                name_cutoff[term] = 1.3
            elif j_el == 'H1':
                name_cutoff[term] = 1.3

        elif i_el == 'Ca':
            if (j_el == 'Si') or (j_el == 'Al'):
                name_cutoff[term] = 1.5
            elif j_el == 'Ca':
                name_cutoff[term] = 1.5
            elif j_el == 'O':
                name_cutoff[term] = 3.0
            elif j_el == 'O1':
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

        elif i_el == 'H1':
            if (j_el == 'Si') or (j_el == 'Al'):
                name_cutoff[term] = 1.81
            elif j_el == 'Ca':
                name_cutoff[term] = 2.5
            elif j_el == 'O':
                name_cutoff[term] = 1.1
            elif j_el == 'O1':
                name_cutoff[term] = 1.1

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
        points.append((x, y, z))

    return(points)

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


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Prepare CAS surface for water interface')
    parser.add_argument('--infile', help='<Required> input file name (data)', required=True)
    args = parser.parse_args(sys.argv[1:])
    # run main
    main(args)
