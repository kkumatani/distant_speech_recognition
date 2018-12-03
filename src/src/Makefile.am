bin_PROGRAMS = superdirectiveBeamformer ss calcTimeDelay

FEATURE_FLAGS = $(GSL_CFLAGS) $(SNDFILE_CFLAGS) $(SAMPLERATE_CFLAGS) $(JACK_CFLAGS) $(OPENCV_CFLAGS) -I$(top_srcdir)/gnulib

superdirectiveBeamformer_SOURCES = superdirectiveBeamformer.cc
superdirectiveBeamformer_LDADD = \
	$(top_srcdir)/common/libbtk_common.la \
	$(top_srcdir)/stream/libbtk_stream.la \
	$(top_srcdir)/feature/libbtk_feature.la \
	$(top_srcdir)/matrix/libbtk_matrix.la \
	$(top_srcdir)/postfilter/libbtk_postfilter.la  \
	$(top_srcdir)/modulated/libbtk_modulated.la \
        $(top_srcdir)/beamformer/libbtk_beamformer.la
superdirectiveBeamformer_CXXFLAGS = $(FEATURE_FLAGS)
superdirectiveBeamformer_LDFLAGS = $(GSL_LIBS) $(GZIP_LIBS) $(SNDFILE_LIBS)

ss_SOURCES = ss.cc
ss_LDADD = \
	$(top_srcdir)/common/libbtk_common.la \
	$(top_srcdir)/stream/libbtk_stream.la \
	$(top_srcdir)/feature/libbtk_feature.la \
	$(top_srcdir)/matrix/libbtk_matrix.la \
	$(top_srcdir)/modulated/libbtk_modulated.la \
	$(top_srcdir)/postfilter/libbtk_postfilter.la
ss_CXXFLAGS = $(FEATURE_FLAGS)
ss_LDFLAGS = $(GSL_LIBS) $(GZIP_LIBS) $(SNDFILE_LIBS)

calcTimeDelay_SOURCES = calcTimeDelay.cc
calcTimeDelay_LDADD = \
	$(top_srcdir)/common/libbtk_common.la \
	$(top_srcdir)/stream/libbtk_stream.la \
	$(top_srcdir)/feature/libbtk_feature.la \
	$(top_srcdir)/matrix/libbtk_matrix.la \
	$(top_srcdir)/TDEstimator/libbtk_TDEstimator.la 
calcTimeDelay_CXXFLAGS = $(FEATURE_FLAGS)
calcTimeDelay_LDFLAGS = $(GSL_LIBS) $(GZIP_LIBS) $(SNDFILE_LIBS)
