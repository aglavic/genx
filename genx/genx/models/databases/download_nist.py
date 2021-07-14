'''
Script to download the FFAST library and convert it to nff files
'''

__author__='Matts Bjorck'

import urllib.request, urllib.parse, urllib.error, numpy, os, time

def get_html_code(Z):
    adress='http://physics.nist.gov/cgi-bin/ffast/ffast.pl?Z=%d&Formula=&gtype=4&lower=&upper=&density=&frames=no'%Z
    f=urllib.request.urlopen(adress)
    return f.readlines()

def find_line(lines, string_to_find):
    for i, line in enumerate(lines):
        if line.find(string_to_find)!=-1:
            return line, i
    return None, None

def parse_page(lines):
    # Find the element
    line, i=find_line(lines, '(Z =')
    element=line.split('<b>')[1].split('&#160;')[0].strip()
    print('Element: ', element)

    # Find the Relativistic correction estimate
    line, i=find_line(lines, 'Relativistic correction estimate')
    rel_corr=float(line.split('=')[1].split(',')[1].split('<i>e</i>')[0].strip())
    print('Relativistic correction estimate 3/5CL: ', rel_corr)

    # Find the Nuclear Thomson correction
    line, i=find_line(lines, 'Nuclear Thomson correction')
    nuc_corr=float(line.split('=')[1].split('<i>e</i>')[0].strip())
    print('Nuclear Thomson correction: ', nuc_corr)

    # Find the data
    line, i=find_line(lines, '<b>Form Factors,')
    data_lines=lines[i+3:-3]
    data=numpy.fromstring(' '.join(data_lines), sep=' ').reshape((len(data_lines), 8))
    table=numpy.c_[data[:, 0]*1e3, data[:, 1]+rel_corr+nuc_corr, data[:, 2]]
    return table, element

def create_nff(Z, path):
    html_code=get_html_code(Z)
    table, element=parse_page(html_code)
    with open(os.path.join(path, '%s.nff'%element.lower()), 'w') as f:
        f.write('E (eV)\tf1 e/atom\tf2 e/atom\n')
        numpy.savetxt(f, table)

def create_readme(path):
    readme_str=(
            '''README
    These tables were generated %s.
    The following files were created by the download_nist.py script located in the GenX distribution. The data
    were generated from the NIST X-ray Form Factor, Attenuation and Scattering Tables (FFAST) located at:
    http://www.nist.gov/pml/data/ffast/index.cfm.
    
    Each element's scattering factors were evaluated according to eq. 4 in
    C. T. Chandler J. Phys. Chem. Ref. Data 29(4) 597-1048 (2000). Thus the scattering factors were calculated
    according to
        f1 = f_1 + f_rel(3/5 CL) + f_NT,
        f2 = f_2.
    
    According to the homepage, see above, the tables should be referenced according to:
        Chantler, C.T., Olsen, K., Dragoset, R.A., Chang, J., Kishore, A.R., Kotochigova, S.A., and Zucker, D.S. (2005),
        X-Ray Form Factor, Attenuation and Scattering Tables (version 2.1).
        [Online] Available: http://physics.nist.gov/ffast [%s].
        National Institute of Standards and Technology, Gaithersburg, MD.
        Originally published as Chantler, C.T., J. Phys. Chem. Ref. Data 29(4), 597-1048 (2000);
        and Chantler, C.T., J. Phys. Chem. Ref. Data 24, 71-643 (1995).
            '''%(time.ctime(), time.ctime())
    )
    with open(os.path.join(path, 'README.txt'), 'w') as f:
        f.write(readme_str)

if __name__=='__main__':
    lib_path='f1f2_nist'
    for z in range(1, 93):
        create_nff(z, lib_path)
    create_readme(lib_path)
