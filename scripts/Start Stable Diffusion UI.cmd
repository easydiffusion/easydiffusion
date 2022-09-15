@call installer\Scripts\activate.bat

@call conda-unpack

@call conda --version
@call git --version

@call scripts\on_env_start.bat

@pause