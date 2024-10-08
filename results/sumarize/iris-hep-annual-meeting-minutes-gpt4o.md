# IRIS-HEP Institute Retreat Meeting Notes - 9/4/2023 - 9/6/2023

## US ATLAS

### Action Items
- Consider guiding hardware purchase decisions (Shawn)
- Target tests around multiple users accessing different data types (Lindsey)
- Assess the feasibility of pushing the 200 Gbps challenge (Peter)

### Decisions Made
- Evolve analysis to be a first-class citizen in regular data processing (Doug)
- Focus experiments on Analysis Facility resources (Rob)

### Big Successes
- N/A

## US CMS

### Action Items
- Engage with the Dask team to provide reproducible failing examples (Matthew)
- Establish a viable support and maintenance structure for Dask (Matthew)
- Relate GPU usage to dollar per use, including cooling, etc. (JimP, Andrew, Lindsey)

### Decisions Made
- Users are split into different Analysis Facilities based on training and some geographical skew (Andrew)

### Big Successes
- N/A

## Coffea State & Integration

### Action Items
- Focus on long-term support model for Coffea
- Investigate a reverse tree reduction mentoring system for codebase familiarity (Jim)
- Consider promoting nanoevents as a standalone package (Lindsey)
- Interface with the Dask community to address task graph optimization issues (Lindsey)
- Set up a server with ASV for continuous benchmarking (Matthew)

### Decisions Made
- Experiment-specific schemas should be considered as plugins to break version coupling (Matthew, Lindsey)
- Establish protocols for optimization flags similar to gcc compiler (Peter, Jim)

### Big Successes
- PHYSLITE format is finalized and frozen (Gordon)
- Postdocs initiated projects in uproot and dask-awkward (Jim)

## OSG Software & Services Needs from LHC Community

### Action Items
- Enable fireflies by default in osg-xrootd and assist sites with deployment
- Collaborate between OSDF and LHC on Pelican proof-of-concept
- Address XCache slowness in deployment and user experience issues

### Decisions Made
- OSG 24 will have full multi-arch support and tell ARM sites to use OSG 24
- HTCondor 24 will remove old job router syntax with automated translation tools

### Big Successes
- EOS@CERN already configured for fireflies
- Pelican successfully tested with XRootD globus backend in "tech preview"

## Training

### Action Items
- Make training milestones measurable and quantify needs
- Conduct focus groups with experienced graduate students and postdocs to determine needed topics
- Determine which topics students are weak in and gather data externally
- Decide on the timing and format of introductory training sessions based on student start times
- Gather data on the number of students from U.S. institutes working on LHC/HL-LHC analyses
- Use certificates as incentives for training participation
- Measure awareness and effectiveness of HSF-Training through surveys
- Consider virtual events for targeting international students

### Decisions Made
- Target training primarily at graduate students from U.S. institutes involved with LHC or HL-LHC

### Big Successes
- N/A

## ServiceX - Status and Futures

### Action Items
- Validate 3.0 beta on AGC before releasing as v3.0.0
- Update and separate production and integration/production instances for testing releases
- Improve monitoring and provide regular smoke tests for site admins
- Present a simple snapshot of current status to users
- Explore using keycloak for simpler authentication

### Decisions Made
- Dedicate more effort to resolving open tickets for improvements post 3.0 client finalization (Peter)

### Big Successes
- OKD working with BNL or SLAC for atlas OKD distribution

## OSG-LHC

### Action Items
- Investigate Kueue or other Kubernetes scheduler mechanisms for T2s
- Collaborate with CMS to build a common solution for OSG-supported containers and backfill

### Decisions Made
- Proceed with multi-arch support in OSG 24 instead of rebuilding OSG 23 for ARM

### Big Successes
- CMS uses OSG-built container images and is testing Pelican for multi-tenancy

---

Note: Some projects did not have explicit action items, decisions, or big successes mentioned in the provided meeting minutes.
