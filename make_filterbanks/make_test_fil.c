#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

FILE *output;

void generate_random_vector(unsigned char *vector, int length, int seed) {

  srand(seed);

  printf("First few values...\n");
  for (int i = 0; i < length; i++) {
    vector[i] = (unsigned char)(rand() % 256);
    if(i<8) printf("%d: %hhu\n", i, vector[i]);    
  }
}

void send_string(char *string) /* includefile */
{
  int len;
  len=strlen(string);
  fwrite(&len, sizeof(int), 1, output);
  fwrite(string, sizeof(char), len, output);
}

void send_float(char *name,float floating_point) /* includefile */
{
  send_string(name);
  fwrite(&floating_point,sizeof(float),1,output);
}

void send_double (char *name, double double_precision) /* includefile */
{
  send_string(name);
  fwrite(&double_precision,sizeof(double),1,output);
}

void send_int(char *name, int integer) /* includefile */
{
  send_string(name);
  fwrite(&integer,sizeof(int),1,output);
}

void send_char(char *name, char integer) /* includefile */
{
  send_string(name);
  fwrite(&integer,sizeof(char),1,output);
}

void send_long(char *name, long integer) /* includefile */
{
  send_string(name);
  fwrite(&integer,sizeof(long),1,output);
}

void send_coords(double raj, double dej, double az, double za) /*includefile*/
{
  if ((raj != 0.0) || (raj != -1.0)) send_double("src_raj",raj);
  if ((dej != 0.0) || (dej != -1.0)) send_double("src_dej",dej);
  if ((az != 0.0)  || (az != -1.0))  send_double("az_start",az);
  if ((za != 0.0)  || (za != -1.0))  send_double("za_start",za);
}

void usage()
{
  fprintf (stdout,
	   "make_test_fil [options]\n"
	   " -f frequency of first channel (MHz, default 1000)\n"
	   " -c channel width (MHz, default -0.1302, will always be negative)\n"
	   " -n number of channels (default 1000)\n"
	   " -t time sampling (in seconds, default 0.001)\n"
	   " -l length of data (in seconds, default 10)\n"
	   " -o output file name (default test.fil)\n"
	   " -s seed for RNG (default 1234)\n"
	   " -h print usage\n");
}


int main(int argc, char *argv[]) {

  float fch1 = 1000.0;
  float chbw = -0.1302;
  int nch = 1000;
  float tsamp = 0.001;
  float ldata = 10.0;
  char outnam[300] = "test.fil";
  int seed = 1234;
  int arg = 0;
  
  while ((arg=getopt(argc,argv,"f:c:n:t:l:o:s:h")) != -1)
    {
      switch (arg)
	{
	case 'f':
	  if (optarg)
	    {
	      fch1 = atof(optarg);
	      break;
	    }
	  else
	    {
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'c':
	  if (optarg)
	    {
	      chbw= atof(optarg);
	      break;
	    }
	  else
	    {
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'n':
	  if (optarg)
	    {
	      nch = atoi(optarg);
	      break;
	    }
	  else
	    {
	      usage();
	      return EXIT_FAILURE;
	    }
	case 't':
	  if (optarg)
	    {
	      tsamp = atof(optarg);
	      break;
	    }
	  else
	    {
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'l':
	  if (optarg)
	    {
	      ldata = atof(optarg);
	      break;
	    }
	  else
	    {
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'o':
	  if (optarg)
	    {
	      strcpy(outnam,optarg);
	      break;
	    }
	  else
	    {
	      usage();
	      return EXIT_FAILURE;
	    }
	case 's':
	  if (optarg)
	    {
	      seed = atoi(optarg);
	      break;
	    }
	  else
	    {
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'h':
	  usage();
	  return EXIT_SUCCESS;
	}
    }

  printf("Freq start: %f\n", fch1);
  printf("Bandwidth:  %f\n", chbw);
  printf("N channels: %d\n", nch);
  printf("T samp(s):  %f\n", tsamp);
  printf("T data(s):  %f\n", ldata);
  printf("Output:     %s\n", outnam);
  printf("Seed:       %d\n", seed);
  
  output = fopen(outnam,"wb");

  send_string("HEADER_START");
  send_string("source_name");
  send_string("test");
  send_int("machine_id",1);
  send_int("telescope_id",82);
  send_int("data_type",1); // filterbank data
  send_double("fch1",fch1); // THIS IS CHANNEL 0 :)
  send_double("foff",chbw);
  send_int("nchans",nch);
  send_int("nbits",8);
  send_double("tstart",55000.0);
  send_double("tsamp",tsamp);
  send_int("nifs",1);
  send_string("HEADER_END");

  float nsamps_f = floor(ldata/tsamp);
  int blocksize = 16;
  int nsamps = blocksize*((int)(floor(nsamps_f/(1.*blocksize))));
  int nblocks = nsamps / blocksize;
  unsigned char * block = (unsigned char *)malloc(blocksize*nch*sizeof(unsigned char));
  generate_random_vector(block, blocksize*nch, seed);
  
  for (int i=0;i<nblocks;i++) {
    fwrite(block,1,blocksize*nch,output);
  }

  free(block);
  fclose(output);
}
