import pandapower.networks as nw
import pandapower as pp

net = nw.case33bw()
pp.runpp(net)

print('VS Code 실행 성공! 손실:', round(net.res_line.pl_mw.sum()*1000, 1), 'kW')