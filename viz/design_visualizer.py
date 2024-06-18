"""
Quick n' dirty script for visualizing drone designs using tikz graphs

"""
from data.datamodel.Grammar import UAVGrammar
from data.Constants import *

comp_colors = {
	"0": "green",
	"1": "red",
	"2": "blue",
	"3": "cyan",
	"4": "white"
}


def draw_drone(uav_str, caption="", nodef=False):
	#header = r"\RequirePackage{tikz}" + "\n"
	#header += r"\RequirePackage{pgfplots}" + "\n"
	forbidden_symbols = ['&', '%', '#', '(', ')', '{', '}', '[', ']', '$']
	alt_symbols = ['amp', 'pct', 'hsh', 'lp', 'rp', 'lb', 'rb', 'lsb', 'rsb', 'usd']
	
	if nodef:
		header = r"\tikzstyle{comp} = [circle,fill=white,draw=black,inner sep=0pt,line width=0.6pt, minimum size=18pt]" + "\n"
	else:
		header = "\n"
	header += r"\begin{tikzpicture}" + "\n"
	footer = r"\end{tikzpicture}\\\\"
	
	parser = UAVGrammar()
	components, connections, payload, _ = parser.parse(uav_str)
	
	body = ""
	comp_id_node_id = {}
	node_idx = 1
	for comp in components:
		comp_id = comp[0]

		if comp_id in forbidden_symbols:
			comp_id = alt_symbols[forbidden_symbols.index(comp_id)]
			
		comp_id_node_id[comp_id] = f"n{node_idx}"
			
		x, z = X_LETTER_COORDS.index(comp[1]), Z_LETTER_COORDS.index(comp[2])
		node_str = r"\node[comp, " + f"fill={comp_colors[comp[3]]}] " +  f"({comp_id_node_id[comp_id]}) at ({x},{z}) " + "{$" + comp_id + "$};\n"
		body += node_str
		node_idx += 1
		
	for conn in connections:
		inp, outp = conn[0], conn[1]
		
		if inp in forbidden_symbols:
			inp = alt_symbols[forbidden_symbols.index(inp)]
			
		if outp in forbidden_symbols:
			outp = alt_symbols[forbidden_symbols.index(outp)]
		
		conn_str = f"\draw[] " + f"({comp_id_node_id[inp]}) -- node[]" + "{}" + f" ({comp_id_node_id[outp]});\n"
		body += conn_str
		
	return header + body + footer


if __name__ == "__main__":
	drone_graph = draw_drone("*aMM0---*bNM2+++++*cLM1+++++*dOM0*eKM0*fPM1*gJM1*hON4*iKL4*jOL4*kKN4*lOK0*mKO0*nOO4*oKK4*pOJ4*qKP4*rNO4*sLK4*tPL2*uJN2*vOP2*wKJ2*xPP0*yJJ0*zPO3*!JK3*@NL4*#LN4*$PJ4*%JP4*&PK2*(JO2*)PN0*_JL0*=NK0*[LO0*]NJ3*{LP3*}NP1*<LJ1^ab^ac^bd^ce^df^eg^dh^ei^dj^ek^jl^km^hn^io^lp^mq^nr^os^jt^ku^ft^gu^nv^ow^vx^wy^nz^o!^xz^y!^j@^k#^b@^c#^p$^q%^l&^m(^$&^%(^t&^u(^h)^i_^f)^g_^z)^!_^l=^m[^@=^#[^p]^q{^=]^[{^v}^w<^r}^s<,5,3")
	with open("test.tikz", "w") as f:
		f.write(drone_graph)