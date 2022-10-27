import matplotlib.pyplot as plt
import numpy as np

def single_trial_plots(condition,trial,a_record,a_tilde_record,formant_record,perturb_record,x_record,argv):
    gest_name = argv[1].split('/')[-1]
       
    plt.figure(condition + 'articulator state')
    plt.plot(a_tilde_record[trial,:,:],marker='o',markersize=3,linestyle='None')
    plt.gca().set_prop_cycle(None) 
    plt.plot(a_record[trial,:,:])
    plt.gca().set_prop_cycle(None) 
    plt.legend(['jaw','tongue','shape','apex','lip_ht','lip_pr'])
    plt.xlabel('frame #')
    plt.ylabel('position')
    plt.title('Configs: ' + argv[0] + ', GesturalScore: ' + gest_name)#, pad=20)
    #plt.tight_layout()
    
    plt.figure(condition + 'formants')
    plt.plot(formant_record[trial,:,0])
    plt.gca().set_prop_cycle(None) 
    plt.plot(perturb_record[trial,:,0])
    plt.gca().set_prop_cycle(None) 
    plt.legend(['F1'])
    plt.xlabel('frame #')
    plt.ylabel('freq (Hz)')
    plt.title('Configs: ' + argv[0] + ', GesturalScore: ' + gest_name)
    
    plt.figure(condition + 'task state')
    plt.plot(x_record[trial])
    plt.gca().set_prop_cycle(None) 
    plt.legend(['TT_Den','TT_Alv','TB_Pal','TB_Vel','TB_Pha','LA','LPRO'],fontsize=12)
    plt.xlabel('Time',fontsize=12)
    plt.ylabel('Constriction Degree(mm)',fontsize=12)
    plt.title('Task Parameters',fontsize=14)
    plt.yticks(fontsize=12)
    #plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    #plt.title('Configs: ' + argv[0] + ', GesturalScore: ' + gest_name)
    
def multi_trial_plots(formants, formants_perturbed):
    plt.figure()
    plt.gca().set_prop_cycle(None) 
    plt.plot(np.median(formants[:,:,:],axis=1))
    plt.gca().set_prop_cycle(None) 
    plt.plot(np.median(formants[:,:,:],axis=1))
    plt.xlabel('trial #')
    plt.ylabel('freq (Hz)')

def smc_abstract_figure(condition,trial,a_record,formant_record,perturb_record,x_record,argv):
    #gest_name = argv[1].split('/')[-1]
    fontsize=20
    plt.figure(figsize=(20,10))
    ax1 = plt.subplot(2,2,1)
    plt.text(-0.15, 1.2, 'A', fontsize=fontsize*2,va='top',ha='right',transform=ax1.transAxes)
    plt.axvspan(100, 400, color='gray',alpha=0.2)
    plt.plot(perturb_record[0,:,0],color='blue',linewidth=5)
    plt.plot(formant_record[0,:,0],color='orange',linewidth=5)
    plt.xlabel('Time', fontsize=fontsize+4)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.ylabel('F1 (Hz)',fontsize=fontsize+4)
    plt.yticks(fontsize=fontsize)
    plt.annotate('perceived F1',(200,560),color='blue',fontsize=fontsize+4)
    plt.annotate('produced F1',(200,410),color='orange',fontsize=fontsize+4)
    plt.ylim([380,580])
    plt.annotate('Pert. on',(100,385),fontsize=fontsize)
    
    ax2 = plt.subplot(2,2,2)
    ax2.text(-0.15, 1.2, 'B', fontsize=fontsize*2,va='top',ha='right',transform=ax2.transAxes)
    x_record_actual = np.loadtxt('x_record.csv',delimiter=',')
    plt.plot(x_record_actual,linewidth=5)
    plt.gca().set_prop_cycle(None) 
    plt.legend(['TT_Den','TT_Alv','TB_Pal','TB_Vel','TB_Pha','LA','LPRO'],bbox_to_anchor=(1.01, 0.95),fontsize=fontsize)
    plt.xlabel('Time',fontsize=fontsize+4)
    plt.ylabel('Constriction Degree\n(mm)',fontsize=fontsize+4)
    plt.yticks(fontsize=fontsize)
    plt.axvspan(100, 400, color='gray',alpha=0.2)
    plt.ylim([11,33])
    plt.annotate('Pert. on',(100,11),fontsize=fontsize)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        
    ax3 = plt.subplot(2,4,5)
    ax3.text(-0.15, 1.2, 'C', fontsize=fontsize*2,va='top',ha='right',transform=ax3.transAxes)
    plt.axvspan(100, 400, color='gray',alpha=0.2)
    plt.axhline(0,0,500,color='black')
    plt.plot(a_record[0,:,1],color='red',linewidth=5)
    plt.title('Tongue',fontsize=fontsize+4)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.annotate('Pert. on',(100,-0.24),fontsize=fontsize)
    plt.ylim([-0.24,0.07])
    plt.ylabel('Articulator Position \n(St.Dev. from mean)',fontsize=fontsize+4)
    plt.xlabel('Time',fontsize=fontsize+4)
    plt.yticks(fontsize=fontsize)
    
    ax4 = plt.subplot(2,4,6)
    ax4.text(-0.15, 1.2, 'D', fontsize=fontsize*2,va='top',ha='right',transform=ax4.transAxes)
    plt.axvspan(100, 400, color='gray',alpha=0.2)
    plt.axhline(0,0,500,color='black')
    plt.plot(a_record[0,:,2],color='red',linewidth=5)
    plt.title('Shape',fontsize=fontsize+4)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.annotate('Pert. on',(100,-0.1),fontsize=fontsize)
    plt.ylim([-0.1,0.2])
    plt.yticks(fontsize=fontsize)
    plt.xlabel('Time',fontsize=fontsize+4)
    
    ax5 = plt.subplot(2,4,7)
    ax5.text(-0.15, 1.2, 'E', fontsize=fontsize*2,va='top',ha='right',transform=ax5.transAxes)
    plt.axvspan(100, 400, color='gray',alpha=0.2)
    plt.axhline(0,0,500,color='black')
    plt.plot(a_record[0,:,0],color='red',linewidth=5)
    plt.title('Jaw',fontsize=fontsize+4)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.annotate('Pert. on',(100,-0.1),fontsize=fontsize)
    plt.ylim([-0.1,0.55])
    plt.yticks(fontsize=fontsize)
    plt.xlabel('Time',fontsize=fontsize+4)
    
    ax6 = plt.subplot(2,4,8)
    ax6.text(-0.15, 1.2, 'F', fontsize=fontsize*2,va='top',ha='right',transform=ax6.transAxes)
    plt.axvspan(100, 400, color='gray',alpha=0.2)
    plt.axhline(0,0,500,color='black')
    plt.plot(a_record[0,:,3],color='red',linewidth=5)
    plt.title('Apex',fontsize=fontsize+4)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.annotate('Pert. on',(100,-0.4),fontsize=fontsize)
    plt.ylim([-0.4, 0.1])
    plt.yticks(fontsize=fontsize)
    plt.xlabel('Time',fontsize=fontsize+4)
    plt.tight_layout()
    plt.show()
