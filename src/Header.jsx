function Header() {
  return (
    <header className="bg-indigo-600 text-white shadow-sm">
      <div className="container mx-auto px-4 py-3 mb-8">
        <div className="text-center">
          <h1 className="text-4xl font-semibold md:text-3xl">
            Skill Gap Analyzer
          </h1>
          <p className="mt-1 text-indigo-100 text-lg md:text-base">
            Find the missing skills for your dream job
          </p>
        </div>
      </div>
    </header>
  );
}

export default Header;