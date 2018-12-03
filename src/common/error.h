#ifdef __cplusplus
extern "C" {
#endif

#ifndef ERROR_H
#define ERROR_H

#define INFO     msg_handler_ptr(__FILE__,__LINE__,-1,0)
#define WARN     msg_handler_ptr(__FILE__,__LINE__,-2,0)
#define SWARN    msg_handler_ptr(__FILE__,__LINE__,-2,1)
#define ERROR    msg_handler_ptr(__FILE__,__LINE__,-3,0)
#define SERROR   msg_handler_ptr(__FILE__,__LINE__,-3,1)
#define FATAL    msg_handler_ptr(__FILE__,__LINE__,-4,0)

#define MSGCLEAR msg_handler_ptr(__FILE__,__LINE__,0,2)
#define MSGPRINT msg_handler_ptr(__FILE__,__LINE__,0,3)

typedef int MsgHandler(   char*, ... );

extern MsgHandler*   msg_handler_ptr(char* file, int line, int type, int mode);
extern int           error_msg_handler(   char*, ... );
extern int           init_error_handler(void);
extern char*         get_error_msg(void);

#endif



#ifdef __cplusplus
}
#endif
