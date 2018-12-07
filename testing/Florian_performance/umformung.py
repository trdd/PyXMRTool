import os

source_folder="Experiment"
dest_folder="Exp_Umgeformt"

files = [source_folder+"/"+name for name in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, name)) and not name==".directory"]
files.sort()

energies=[]
for item in files:
  name=item.split("/")[-1]
  name=name[:-4]
  name_splitted=name.split("_")
  energy=float(name_splitted[-2])
  energies.append(energy)
  
refl=dict(zip(energies,[[None,None,None] for item in energies]))

for item in files:
  name=item.split("/")[-1]
  name=name[:-4]
  name_splitted=name.split("_")
  energy=float(name_splitted[-2])
  polarization=name_splitted[-1]
  with open(item,"r") as f:
    lines=f.readlines()
  qs=[]
  rs=[]
  for line in lines:
    line=line.split()
    qs.append(float(line[0]))
    rs.append(float(line[1]))
  refl[energy][0]=qs
  if polarization=="sigma":
    refl[energy][1]=rs
  elif polarization=="pi":
    refl[energy][2]=rs

for energy in refl.keys():
  lines=zip(refl[energy][0],refl[energy][1],refl[energy][2])
  with open(dest_folder+"/sro_lsmo_"+str(energy)+".dat","w") as f:
    for line in lines:
      line=[str(item) for item in line]
      line=" ".join(line)+"\n"
      f.write(line)


