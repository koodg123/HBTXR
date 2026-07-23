#ifndef __INT_UTILS_H__
#define __INT_UTILS_H__


template<typename _typename, typename _val_t>
_typename clamp(_typename val, _val_t min_val, _val_t max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

template<typename _typename>
_typename quantize_clamp(_typename val, int bits, bool is_signed) {
    int min_val, max_val;
    if (is_signed) {
        min_val = -(1 << (bits - 1));
        max_val = +(1 << (bits - 1)) - 1;
    } else {
        min_val = 0;
        max_val = (1 << bits) - 1;
    }
    return clamp(val, min_val, max_val);
}


#ifndef __SYNTHESIS__

template<class __data_t, int _T, int _TP, int _C, int _CP>
void check_stream(hls::stream<hls::vector<__data_t, _CP*_TP> >& stream, const int ref[_T*_C], const char* name){
    constexpr int _TT = _T / _TP;
    constexpr int _CT = _C / _CP;

    for(int tt = 0; tt < _TT; tt++){
        for(int ct = 0; ct < _CT; ++ct){
            hls::vector<__data_t, _TP*_CP> vec = stream.read();
            for(int tp = 0; tp < _TP; tp++){
                for(int cp = 0; cp < _CP; cp++){
                    int t = tt * _TP + tp;
                    int c = ct * _CP + cp;
                    int idx = t*_C + c;
                    int dut_val = vec[tp*_CP + cp];
                    int ref_val = ref[idx];
                    if(dut_val != ref_val){
                        printf("ERROR at %s: stream[%d][%d] = %d, ref[%d] = %d\n", name, tt, ct, dut_val, idx, ref_val);
                    }
                }
            }
        }
    }

    if(!stream.empty()){
        cout << "ERROR: in checking " << name << " stream, stream is not empty!" << endl;
        exit(1);
    }

    // write back
    for(int tt = 0; tt < _TT; tt++){
        for(int ct = 0; ct < _CT; ++ct){
            hls::vector<__data_t, _TP*_CP> vec;
            for(int tp = 0; tp < _TP; tp++){
                for(int cp = 0; cp < _CP; cp++){
                    int t = tt * _TP + tp;
                    int c = ct * _CP + cp;
                    int idx = t*_C + c;
                    vec[tp*_CP + cp] = ref[idx];
                }
            }
            stream.write(vec);
        }
    }
}

#endif


#endif
