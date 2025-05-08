function Footer() {
    return (
      <footer className="bg-black text-white py-6 mt-12">
        <div className="container mx-auto px-4 text-center">
          <p>&copy; {new Date().getFullYear()} Joyce Chapi. All rights reserved.</p>
          <p className="mt-2 text-gray-400 text-sm">Helping you bridge the skill gap</p>
        </div>
      </footer>
    );
  }
  
  export default Footer;