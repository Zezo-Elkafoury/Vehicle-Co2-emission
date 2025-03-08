import Link from "next/link"
import { Leaf, Mail, Github, Twitter } from "lucide-react"

export default function Footer() {
  return (
    <footer className="bg-gradient-to-r from-green-700 to-blue-700 text-white py-16 relative overflow-hidden">
      {/* Animated background */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-0 left-0 w-full h-full opacity-10">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 320" className="absolute top-0">
            <path
              fill="#ffffff"
              fillOpacity="1"
              d="M0,224L48,213.3C96,203,192,181,288,181.3C384,181,480,203,576,202.7C672,203,768,181,864,186.7C960,192,1056,224,1152,218.7C1248,213,1344,171,1392,149.3L1440,128L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"
            ></path>
          </svg>
        </div>
      </div>

      <div className="container mx-auto px-4 relative z-10">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
          <div className="animate-fadeIn">
            <div className="flex items-center space-x-2 mb-6">
              <div className="bg-white rounded-full p-2">
                <Leaf className="h-6 w-6 text-green-600" />
              </div>
              <span className="font-bold text-2xl">EcoLearn</span>
            </div>
            <p className="text-green-100 mb-6">
              Our Enviroment, Our life
            </p>
           
          </div>

          <div className="animate-fadeIn" style={{ animationDelay: "0.2s" }}>
            <h3 className="font-semibold text-xl mb-6 border-b border-green-600 pb-2">Quick Links</h3>
            <ul className="space-y-3">
              <li>
                <Link href="/" className="text-green-100 hover:text-white transition-colors flex items-center">
                  <span className="bg-green-600 h-1.5 w-1.5 rounded-full mr-2"></span>
                  Home
                </Link>
              </li>
              <li>
                <Link href="/content" className="text-green-100 hover:text-white transition-colors flex items-center">
                  <span className="bg-green-600 h-1.5 w-1.5 rounded-full mr-2"></span>
                  Educational Content
                </Link>
              </li>
              <li>
                <Link href="/calc" className="text-green-100 hover:text-white transition-colors flex items-center">
                  <span className="bg-green-600 h-1.5 w-1.5 rounded-full mr-2"></span>
                  CO2 Calculator
                </Link>
              </li>
              <li>
                <Link
                  href="/eco-tracker"
                  className="text-green-100 hover:text-white transition-colors flex items-center"
                >
                  <span className="bg-green-600 h-1.5 w-1.5 rounded-full mr-2"></span>
                  Eco Tracker
                </Link>
              </li>
            </ul>
          </div>

          <div className="animate-fadeIn" style={{ animationDelay: "0.4s" }}>
            
            <ul className="space-y-3">
              
            
    
            </ul>
          </div>
        </div>

        <div className="border-t border-green-600/50 mt-12 pt-8 text-center text-green-100">
          <p>&copy; {new Date().getFullYear()} EcoLearn by Ziad Elkafoury. All rights reserved.</p>
        </div>
      </div>
    </footer>
  )
}

