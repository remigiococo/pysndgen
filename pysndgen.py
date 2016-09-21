'''
  Copyright {2016} {Remigio Coco}

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
'''

from audioop import * 
import wave
import random
import struct
from math import *
from array import *
from pmask import ScoreSection
from pmask.exception import *

# -----------------------------------------------------------------------------	
# generators
# -----------------------------------------------------------------------------	
def gen_silence(secondi, stereo=False, sr=44100) :
	nsamp = int(secondi * sr)
	# 'h' = signed short
	ret = array('h')
	for i in range(nsamp) :
		ret.append( 0 )
		if stereo :
			ret.append( 0 )
	return ret

# (monophonic generators)
def gen_sinus(freq, amp, secondi, sr=44100, ph=0.0) :
	nsamp = int(secondi * sr)
	#ph = 0.0
	dph = 2.0 * pi * freq / sr
	# 'h' = signed short
	ret = array('h')
	for i in range(nsamp) :
		ret.append( int(sin(ph) * amp * 32767.0) )
		ph = ph + dph
		if( ph > 2.0 * pi ) :
			ph = ph - 2.0 * pi
	return ret

def gen_pulsar(freq, amp, secondi, duty=0.5, sr=44100) :
	nsamp = int(secondi * sr)
	ph = 0.0
	dph = 2.0 * pi * freq / sr
	# phase correction factor
	phcorr = 1.0 / duty
	# 'h' = signed short
	ret = array('h')
	for i in range(nsamp) :
		if( ph >= 2.0 * pi * duty ) :
			ret.append(0)
		else :	
			ret.append( int(sin(ph * phcorr) * amp * 32767.0) )
		ph = ph + dph
		if( ph > 2.0 * pi ) :
			ph = ph - 2.0 * pi
	return ret

def gen_pulsarenv(freq, amp, secondi, duty=0.5, envtype='tri', sr=44100) :
	nsamp = int(secondi * sr)
	ph = 0.0
	dph = 2.0 * pi * freq / sr
	# phase correction factor
	phcorr = 1.0 / duty
	# 'h' = signed short
	ret = array('h')
	for i in range(nsamp) :
		if( envtype == 'tri' ) :
			# triang. envelope
			env = 1.0 - fabs(1.0- 2.0*float(i)/float(nsamp))
		elif( envtype == 'sawd' ) :
			env = 1.0 - float(i)/float(nsamp)
		elif( envtype == 'sawu' ) :
			env = float(i)/float(nsamp)
		else :
			env = 1.0
		if( ph >= 2.0 * pi * duty ) :
			ret.append(0)
		else :	
			ret.append( int(sin(ph * phcorr) * amp * env * 32767.0) )
		ph = ph + dph
		if( ph > 2.0 * pi ) :
			ph = ph - 2.0 * pi
	return ret
	
def gen_sinusenv(freq, amp, secondi, envtype='tri', sr=44100, ph=0.0) :
	nsamp = int(secondi * sr)
	#ph = 0.0
	dph = 2.0 * pi * freq / sr
	# 'h' = signed short
	ret = array('h')
	for i in range(nsamp) :
		if( envtype == 'tri' ) :
			# triang. envelope
			env = 1.0 - fabs(1.0- 2.0*float(i)/float(nsamp))
		elif( envtype == 'sawd' ) :
			env = 1.0 - float(i)/float(nsamp)
		elif( envtype == 'sawu' ) :
			env = float(i)/float(nsamp)
		elif( envtype == 'noclick' ) :
			NOCLICK_SAMPLES=40
			env = 1
			if i < NOCLICK_SAMPLES:
				env = float(i)/float(NOCLICK_SAMPLES)
			elif i > (nsamp - NOCLICK_SAMPLES):
				env = float(nsamp-i)/float(NOCLICK_SAMPLES)
		else :
			env = 1.0
		ret.append( int(sin(ph) * amp * env * 32767.0) )
		ph = ph + dph
		if( ph > 2.0 * pi ) :
			ph = ph - 2.0 * pi
	return ret

def gen_sinusenv2(freq, amp, secondi, sr=44100) :
	return gen_sinusenv(freq, amp, secondi, 'sawd', sr)

def gen_sinusenv3(freq, amp, secondi, sr=44100) :
	return gen_sinusenv(freq, amp, secondi, 'sawu', sr)
	
def gen_clicks(prob, secondi, sr=44100)	:
	nsamp = int(secondi * sr)
	ret = array('h')
	for i in range(nsamp) :
		r = random.random()
		r2 = 2.0 * random.random() - 1.0
		if( r <= prob ) :
			ret.append( int(32767.0*r2) )
		else :	
			ret.append(0)
	return ret

def gen_ks(freq, a, b, secondi, sr=44100)	:
	# Karplus-Strong algorithm
	tlen = int(sr/freq)
	tab = array('h')
	for i in range(tlen) :
		tab.append( int(random.random() * 32767.0) )
	nsamp = int(secondi * sr)
	ret = array('h')
	for i in range(nsamp-1) :
		j = i % tlen
		s = a * tab[j] + b * tab[(j+1)%tlen]
		tab[j] = int(s)
		ret.append(int(s))
	return ret

def gen_noiseburst(ampiezza, secondi, sr=44100) :
	nsamp = int(secondi * sr)
	ret = array('h')
	for i in range(nsamp) :
		# triang. envelope
		env = 1.0 - fabs(1.0- 2.0*float(i)/float(nsamp))
		ret.append( int( (random.random() * 65534.0 - 32767.0) * env * ampiezza) )
	return ret

def gen_otherburst(ampiezza, secondi, sr=44100) :
	nsamp = int(secondi * sr)
	ret = array('h')
	prevs = 0.0
	for i in range(nsamp) :
		# triang. envelope
		env = 1.0 - fabs(1.0- 2.0*float(i)/float(nsamp))
		s = (random.random() * 65534.0 - 32767.0) * env * ampiezza
		# simple LP filter
		s = 0.1 * s + 0.9 * prevs
		prevs = s
		ret.append( int(s) )
	return ret
# -----------------------------------------------------------------------------	
# granular synthesis - grain_func is a (mono) generator - frequency of single grains is determined by that function
# -----------------------------------------------------------------------------	
def grain1(gdur, gamp, gdens, secondi, grain_func=gen_noiseburst, sr=44100) :
	nsamp = int(secondi * sr)
	ret = array('h')
	ng = gdens * secondi
	print "num. of grains=", ng 
	for i in range(nsamp) :
		ret.append(0)
	for i in range(ng) :	
		v = grain_func(gamp, gdur)
		offset = int(random.random() * secondi * sr)
		#print offset
		mix(ret, v, offset)
	return ret	

# equi-spaced grains	
def train1(gdur, gamp, tfreq, secondi, grain_func=gen_noiseburst, sr=44100) :
	nsamp = int(secondi * sr)
	ret = array('h')
	ng = tfreq * secondi
	print "num. of trains=", ng 
	for i in range(nsamp) :
		ret.append(0)
	for i in range(ng) :	
		v = grain_func(gamp, gdur)
		offset = int(i * sr / tfreq)
		#print offset
		mix(ret, v, offset)
	return ret	

# equi-spaced grains w/ intermittency	
def train1_masked(gdur, gamp, tfreq, prob, secondi, grain_func=gen_noiseburst, sr=44100) :
	nsamp = int(secondi * sr)
	ret = array('h')
	ng = tfreq * secondi
	print "num. of trains=", ng 
	for i in range(nsamp) :
		ret.append(0)
	for i in range(ng) :	
		v = grain_func(gamp, gdur)
		offset = int(i * sr / tfreq)
		#print offset
		if( random.random() < prob ) :
			mix(ret, v, offset)
	return ret	
# -----------------------------------------------------------------------------	
# train2, grain2, etc... -> single grain has a frequency
# -----------------------------------------------------------------------------	
def train2_masked(gdur, gamp, gfreq1, gfreq2, tfreq, prob, secondi, grain_func=gen_sinusenv, sr=44100) :
	nsamp = int(secondi * sr)
	ret = array('h')
	ng = tfreq * secondi
	print "num. of trains=", ng 
	for i in range(nsamp) :
		ret.append(0)
	for i in range(ng) :
		f = random.random() * (gfreq2-gfreq1) + gfreq1	
		v = grain_func(f, gamp, gdur)
		offset = int(i * sr / tfreq)
		#print offset
		if( random.random() < prob ) :
			mix(ret, v, offset)
	return ret	
	
def grain2(gdur, gamp, gdens, gfreq1, gfreq2, secondi, grain_func=gen_sinusenv, sr=44100) :
	nsamp = int(secondi * sr)
	ret = array('h')
	ng = int( gdens * secondi )
	print "num. of grains=", ng 
	for i in range(nsamp) :
		ret.append(0)
	for i in range(ng) :
		f = random.random() * (gfreq2-gfreq1) + gfreq1
		v = grain_func(f, gamp, gdur)
		offset = int(random.random() * secondi * sr)
		#print offset
		mix(ret, v, offset)
	return ret	

# TODO: start_time list	
def gen_grain2_group(gdurlist, amplist, denslist, f1list, f2list, durlist, grain_func=gen_sinusenv, sr=44100) :
	lendur = len(durlist)
	lengdur = len(gdurlist)
	lenamp = len(amplist)
	lendens = len(denslist)
	lenf1 = len(f1list)
	lenf2 = len(f2list)
	if( (lendur!=lenamp) or (lendur!=lendens) or (lendur!=lenf1) or (lendur!=lenf2) or (lendur!=lengdur) ) :
		print "gen_grain2_group error: lengths must be all the same"
		return array('h')
	secondi = 0	
	for i in range(lendur) :	
		secondi = secondi + durlist[i]
	nsamp = int(secondi * sr)
	ret = array('h')
	# main loop
	for i in range(lendur) :
		tmp = grain2(gdurlist[i], amplist[i], denslist[i], f1list[i], f2list[i], durlist[i], grain_func, sr)
		ret.extend(tmp)
	return ret

# -----------------------------------------------------------------------------	
# renderization of "instruments" with parameter lists (max 12 params)
# -----------------------------------------------------------------------------	
global empty_list
empty_list = array('h')

# *p for missing parameters
def instr1(p3v, p4v, p5v, *p) :
	# p3=dur. p4=ampl. p5=freq.
	return gen_sinusenv(p5v, p4v, p3v)

def instr_pulsar(p3v, p4v, p5v, p6v, p7v, p8v, *p) :
	envtype = 'tri'
	if(p6v < 0.2) :
		envtype = 'sawd'
	if(p6v > 0.8) :
		envtype = 'sawu'
	#gen_pulsarenv(freq, amp, secondi, duty=0.5, envtype='tri', sr=44100)
	return gen_pulsarenv(p5v, p4v, p3v, p8v, envtype)

def instr_pulsar_stereo(p3v, p4v, p5v, p6v, p7v, p8v, *p) :
	envtype = 'tri'
	if(p6v < 0.2) :
		envtype = 'sawd'
	if(p6v > 0.8) :
		envtype = 'sawu'
	#gen_pulsarenv(freq, amp, secondi, duty=0.5, envtype='tri', sr=44100)
	x = gen_pulsarenv(p5v, p4v, p3v, p8v, envtype)
	v = array('h')
	for i in range( len(x) ):
		v.append( x[i] * p7v )
		v.append( x[i] * (1.0-p7v) )
	return v
	
# p2 = absolute time (1st element should be zero)
# p4 = 0..32767
def render_instrument(p2, p3, p4, p5, 
p6=empty_list, p7=empty_list, p8=empty_list, p9=empty_list, p10=empty_list,
p11=empty_list, p12=empty_list, ifunc=instr1, channels=1) :
	maxp2 = 0.0
	maxi = 0
	ampfact = 1.0/32767.0
	for i in range( len(p2) ) :
		if p2[i] > maxp2 :
			maxp2 = p2[i]
			maxi = i
	dur = maxp2 + p3[maxi]
	if channels == 1:
		ret = gen_silence(dur)	
	else:	
		ret = gen_silence(dur, stereo=True)
	f6 = (len(p6) == len(p2))
	f7 = (len(p7) == len(p2))
	f8 = (len(p8) == len(p2))
	f9 = (len(p9) == len(p2))
	f10 = (len(p10) == len(p2))
	f11 = (len(p11) == len(p2))
	f12 = (len(p12) == len(p2))
	for i in range( len(p2) ) :
		if f6 : v6 = p6[i] 
		else : v6 = 0.0
		if f7 : v7 = p7[i] 
		else : v7 = 0.0
		if f8 : v8 = p8[i] 
		else : v8 = 0.0
		if f9 : v9 = p9[i] 
		else : v9 = 0.0
		if f10 : v10 = p10[i] 
		else : v10 = 0.0
		if f11 : v11 = p11[i] 
		else : v11 = 0.0
		if f12 : v12 = p12[i] 
		else : v12 = 0.0
		v = ifunc(p3[i], p4[i]*ampfact, p5[i], v6, v7, v8, v9, v10, v11, v12)
		mix(ret, v, sec2smp(p2[i])*channels)
		print "mixing event n. ", i, " p2: ", p2[i] 
	return ret

def get_p(section, index) :
	if not isinstance(section, ScoreSection) :
		raise BadArgument
	ret = []	
	for i in range( len(section) ) :
		# pfields index is 0 for score param p1, 1 for p2, 2 for p3, ... n-1 for pn
		if( len(section[i].pfields) >= index ) :
			pfield = section[i].pfields[index-1]
			ret.append(pfield)
	return ret	
	
def render_score(sc, ifunc=instr1, channels=1) :	
	p2v  = get_p(sc,2)
	p3v  = get_p(sc,3)
	p4v  = get_p(sc,4)
	p5v  = get_p(sc,5)
	p6v  = get_p(sc,6)
	p7v  = get_p(sc,7)
	p8v  = get_p(sc,8)
	p9v  = get_p(sc,9)
	p10v = get_p(sc,10)
	p11v = get_p(sc,11)
	p12v = get_p(sc,12)
	return render_instrument(p2v, p3v, p4v, p5v, p6v, p7v, p8v, p9v, p10v, p11v, p12v, ifunc, channels)
	
# -----------------------------------------------------------------------------	
# transformations
# -----------------------------------------------------------------------------	
def limit_16bit(a) :
	ret = a
	if( ret > 32767 ) :
		ret = 32767
	if( ret < -32767 ) :
		ret = -32767
	return ret

def sec2smp(a, sr=44100) :
	return int(a*sr)

def freq2smp(a, sr=44100) :
	return int(sr/a)

# feedback comb filter / echo (delay in samples)	
def comb(data, dly, feedbk) :
	nsamp = len(data)
	ret = array('h')
	os = 0.0
	for i in range(nsamp) :
		s = data[i]
		if( i >= dly ) :	
			os =  int( s + feedbk * ret[i-dly])
			os = limit_16bit( os )
		else : 
			os = limit_16bit( int(s) )
		ret.append(os)
	return ret	

# mix 2 waves (offset in samples)	
# result is stored in the first vector v1
# if offset + length(v2) > length(v1), v1 is extended 
def mix(v1, v2, offset=0)	:
	len1 = len(v1)
	len2 = len(v2)
	for i in range(len2) :
		if( (i+offset) < len1 ) :
			v = v1[i+offset] + v2[i]
			v1[i+offset] = limit_16bit( v )
		else :
			# v1 is extended
			v1.append(v2[i])

# multiplication between waves (both vectors range from -32768 to 32767)
# if one of the vectors is longer, it's truncated
def mult(v1, v2) :
	len1 = len(v1)
	len2 = len(v2)
	len = min(len1, len2)
	v = array('h')
	for i in range(len) :
		#normalized value
		nv = float(v2[i]) / 32767.0
		s = float(v1[i]) * nv
		v.append(limit_16bit( int(s) ))

# filters (biquad)		
def bp_filter(data, cutoff, Q, sr=44100) :
	w0 = 2.0*pi*cutoff/sr
	alpha = sin(w0)/(2.0*Q)
	b0 = alpha
	b1 = 0.0
	b2 = -alpha
	a0 = 1.0 + alpha
	a1 = -2.0*cos(w0)
	a2 = 1.0 - alpha
	x1 = 0.0
	x2 = 0.0
	y1 = 0.0
	y2 = 0.0
	v = array('h')
	n = len(data)
  #y[n] = (b0/a0)*x[n] + (b1/a0)*x[n-1] + (b2/a0)*x[n-2] - (a1/a0)*y[n-1] - (a2/a0)*y[n-2]
	for i in range(n) :
		x = float(data[i])
		y = (b0/a0)*x + (b1/a0)*x1 + (b2/a0)*x2 - (a1/a0)*y1 - (a2/a0)*y2
		v.append( limit_16bit( int(y) ) )
		x2 = x1
		x1 = x
		y2 = y1		
		y1 = y
	return v

def lp_filter(data, cutoff, Q, sr=44100) :
	w0 = 2.0*pi*cutoff/sr
	alpha = sin(w0)/(2.0*Q)
	b0 =(1.0 - cos(w0))/2.0
	b1 = 1.0 - cos(w0)
	b2 =(1.0 - cos(w0))/2.0
	a0 = 1.0 + alpha
	a1 =-2.0*cos(w0)
	a2 = 1.0 - alpha
	x1 = 0.0
	x2 = 0.0
	y1 = 0.0
	y2 = 0.0
	v = array('h')
	n = len(data)
  #y[n] = (b0/a0)*x[n] + (b1/a0)*x[n-1] + (b2/a0)*x[n-2] - (a1/a0)*y[n-1] - (a2/a0)*y[n-2]
	for i in range(n) :
		x = float(data[i])
		y = (b0/a0)*x + (b1/a0)*x1 + (b2/a0)*x2 - (a1/a0)*y1 - (a2/a0)*y2
		v.append( limit_16bit( int(y) ) )
		x2 = x1
		x1 = x
		y2 = y1		
		y1 = y
	return v

def hp_filter(data, cutoff, Q, sr=44100) :
	w0 = 2.0*pi*cutoff/sr
	alpha = sin(w0)/(2.0*Q)
	b0 =(1.0 + cos(w0))/2.0
	b1 =-(1.0 + cos(w0))
	b2 =(1.0 + cos(w0))/2.0
	a0 = 1.0 + alpha
	a1 =-2.0*cos(w0)
	a2 = 1.0 - alpha
	x1 = 0.0
	x2 = 0.0
	y1 = 0.0
	y2 = 0.0
	v = array('h')
	n = len(data)
  #y[n] = (b0/a0)*x[n] + (b1/a0)*x[n-1] + (b2/a0)*x[n-2] - (a1/a0)*y[n-1] - (a2/a0)*y[n-2]
	for i in range(n) :
		x = float(data[i])
		y = (b0/a0)*x + (b1/a0)*x1 + (b2/a0)*x2 - (a1/a0)*y1 - (a2/a0)*y2
		v.append( limit_16bit( int(y) ) )
		x2 = x1
		x1 = x
		y2 = y1		
		y1 = y
	return v

		
# -----------------------------------------------------------------------------		
# file I/O
# -----------------------------------------------------------------------------	
def save_wave(nome, data, numchannels=1) :
	dlen = len(data)
	#print(dlen)
	dlen = dlen * 2
	#print(dlen)
	w = wave.open(nome, 'w')
	w.setnchannels(numchannels)
	w.setsampwidth(2)
	w.setframerate(44100)
	w.setcomptype('NONE', 'not compressed')
	w.writeframes(data)
	w.close()
	# fix: la writeframes scrive valori sbagliati dei chunk size
	ff = file(nome, 'r+b')
	ff.seek(4, 0)
	ff.write(struct.pack('l', dlen+36))
	ff.seek(40, 0)
	ff.write(struct.pack('l', dlen))
	ff.close()
	
def append_wave(wavefile, data) :
	wavefile.writeframes(data)

def fix_wave_header(nomefile) :
	ff = file(nomefile, 'r+b')
	ff.seek(0, 2)
	dlen = ff.tell()
	dlen = dlen - 44
	ff.seek(4, 0)
	ff.write(struct.pack('l', dlen+36))
	ff.seek(40, 0)
	ff.write(struct.pack('l', dlen))
	ff.close()
# -----------------------------------------------------------------------------		
# TESTS	
# -----------------------------------------------------------------------------	
def test_grain2_group() :
	gdur = array('f')
	gamp = array('f')
	gdens = array('f')
	f1 = array('f')
	f2 = array('f')
	dur = array('f')
	for i in range(60) :
		gdur.append( random.uniform(0.01, 0.1) )
		gamp.append( 0.2 )
		gdens.append( random.choice( (25, 200) ) )
		f1.append( random.uniform(220, 880) )
		f2.append( 1100 )
		dur.append( 0.5	)
	#print gdens	
	x = gen_grain2_group(gdur, gamp, gdens, f1, f2, dur, gen_sinusenv2)
	save_wave('test.wav', x)

def test_mix() :	
	x1 = gen_sinus(200, 0.1, 2)
	x2 = gen_sinus(300, 0.1, 1)
	x3 = gen_sinus(400, 0.1, 1)
	mix(x1,x2, int(1.0*44100))
	mix(x1,x3, int(1.5*44100))
	save_wave('test.wav', x1)

def test_render() :
	p2 = (0.0, 2.0, 4.0)
	p3 = (2.0, 2.0, 4.5)
	p4 = (5000, 5000, 5000)
	p5 = (22, 44, 66)
	p6 = [0.1 for i in range(len(p2))]
	p7 = [0.5 for i in range(len(p2))]
	p8 = [0.1 for i in range(len(p2))]
	return render_instrument(p2, p3, p4, p5, p6, p7, p8, ifunc=instr_pulsar)
	
	
	
#x = gen_sinus(440, 0.5, 10)

#x = gen_clicks(0.001, 5)
#y = comb(x, 120, 0.7)

# x = gen_ks(440, 0.4, 0.4, 0.2)
# for n in range(10) :
	# y = gen_ks(random.random()*220+220, 0.3, 0.3, 0.2)
	# x.extend(y)

#x = grain1(0.005, 0.3, 40, 5, gen_otherburst)
#y = grain1(0.005, 0.3, 40, 5)
#x.extend(y)
#z = comb(y, 333, 0.8)

#x = grain2(0.015, 0.3, 40, 100, 1000, 5)
#x = gen_grain2_group((0.01, 0.02), (0.5, 0.2), (40, 100), (100, 200), (1000, 250), (5, 5) )

#x = train1(0.005, 0.3, 140, 5)
#save_wave('test.wav', x)

#test_grain2_group()

#test_mix()

#x = train2_masked(0.005, 0.3, 1000, 1500, 100, 0.7, 5)
#y = comb(x, 100, 0.8)
#save_wave('test.wav', y)

#x = gen_noiseburst(1.0, 5)

#x = gen_clicks(0.1, 0.1)
#s = gen_silence(5)
#mix(s,x)
#y = comb(s, sec2smp(0.3), 0.75)

#y = test_render()
#save_wave('test.wav', y)
