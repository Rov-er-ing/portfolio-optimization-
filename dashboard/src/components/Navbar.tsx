"use client";

import React from "react";

interface NavbarProps {
  isConnected?: boolean;
}

const Navbar: React.FC<NavbarProps> = ({ isConnected = false }) => {
  return (
    <nav className="fixed top-0 left-0 right-0 h-16 bg-white/80 backdrop-blur-md border-b border-outline-variant/30 z-50 flex items-center justify-between px-8">
      <div className="flex items-center gap-8">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-xl">M</span>
          </div>
          <span className="text-xl font-h1 tracking-tighter text-primary">MLPO v1.0</span>
        </div>
        <div className="hidden md:flex items-center gap-6">
          <a href="#" className="text-sm font-medium text-primary border-b-2 border-primary py-5">Overview</a>
          <a href="#" className="text-sm font-medium text-foreground/60 hover:text-primary py-5 transition-colors">Analytics</a>
          <a href="#" className="text-sm font-medium text-foreground/60 hover:text-primary py-5 transition-colors">Reporting</a>
        </div>
      </div>
      
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 px-4 py-2 bg-surface-container rounded-full border border-outline-variant/20">
          <div className={`w-2 h-2 rounded-full animate-pulse ${isConnected ? "bg-emerald-500" : "bg-amber-500"}`} />
          <span className="text-[10px] font-bold uppercase tracking-widest text-foreground/60">
            {isConnected ? "API: CONNECTED" : "API: DISCONNECTED"}
          </span>
        </div>
        <button className="p-2 hover:bg-surface-container rounded-full transition-colors">
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
            <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
            <circle cx="12" cy="12" r="3" />
          </svg>
        </button>
      </div>
    </nav>
  );
};

export default Navbar;
